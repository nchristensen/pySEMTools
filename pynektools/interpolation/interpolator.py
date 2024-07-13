""" Contains the interpolator class"""

from itertools import combinations
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from mpi4py import MPI  # for the timer
from .mpi_ops import gather_in_root, scatter_from_root
from .point_interpolator.point_interpolator_factory import get_point_interpolator

NoneType = type(None)

# Variables for debugging
DEBUG = False
TR = 7


class Interpolator:
    """Class that interpolates data from a SEM mesh into a series of points"""

    def __init__(
        self,
        x,
        y,
        z,
        probes,
        comm,
        progress_bar=False,
        point_interpolator_type="single_point_legendre",
        max_pts=128,
        max_elems=1,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.probes = probes

        self.point_interpolator_type = point_interpolator_type
        self.max_pts = max_pts
        self.max_elems = max_elems

        # Determine which point interpolator to use
        self.ei = get_point_interpolator(
            point_interpolator_type, x.shape[1], max_pts=max_pts, max_elems=max_elems
        )

        # Determine which buffer to use
        self.r = self.ei.alloc_result_buffer(dtype="double")
        self.s = self.ei.alloc_result_buffer(dtype="double")
        self.t = self.ei.alloc_result_buffer(dtype="double")
        self.test_interp = self.ei.alloc_result_buffer(dtype="double")

        # Print what you are using
        if comm.Get_rank() == 0 and not isinstance(self.r, int):
        
            try:
                print(f"Using device: {self.r.device}")
            except AttributeError:
                print(f"Using device: cpu")

        self.progress_bar = progress_bar

        # Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.intc) * nelv
        recvbuf = np.zeros((1), np.intc)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

        # define dummy varaibles
        self.probes_rst = None
        self.el_owner = None
        self.glb_el_owner = None
        self.rank_owner = None
        self.err_code = None
        self.test_pattern = None
        self.probe_partition_sendcount = None
        self.probe_coord_partition_sendcount = None
        self.probe_partition = None
        self.probe_rst_partition = None
        self.el_owner_partition = None
        self.glb_el_owner_partition = None
        self.rank_owner_partition = None
        self.err_code_partition = None
        self.test_pattern_partition = None
        self.rank = None
        self.my_probes = None
        self.my_probes_rst = None
        self.my_el_owner = None
        self.my_rank_owner = None
        self.my_err_code = None
        self.my_bbox = None
        self.my_bbox_centroids = None
        self.my_bbox_maxdist = None
        self.my_tree = None
        self.ranks_ive_checked = None
        self.size = None
        self.nelv = None
        self.sendcounts = None
        self.sort_by_rank = None

    def scatter_probes_from_io_rank(self, io_rank, comm):
        """Scatter the probes from the rank that is used to read them - rank0 by default"""
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Check how many probes should be in each rank with a load balanced linear distribution
        probe_partition_sendcount = np.zeros((size), dtype=np.intc)
        if rank == io_rank:
            for i_rank in range(0, size):
                m = self.probes.shape[0]
                pe_rank = i_rank
                pe_size = comm.Get_size()
                # l = np.floor(np.double(m) / np.double(pe_size))
                # rr = np.mod(m, pe_size)
                ip = np.floor(
                    (
                        np.double(m)
                        + np.double(pe_size)
                        - np.double(pe_rank)
                        - np.double(1)
                    )
                    / np.double(pe_size)
                )
                nelv = int(ip)
                # offset_el = int(pe_rank * l + min(pe_rank, rr))
                # n = 3 * nelv
                probe_partition_sendcount[i_rank] = int(nelv)

        comm.Bcast(probe_partition_sendcount, root=io_rank)
        probe_coord_partition_sendcount = (
            probe_partition_sendcount * 3
        )  # Since each probe has 3 coordinates

        # Define some variables that need to be only in the rank that partitions
        if rank == io_rank:
            # Set the necesary arrays for identification of point
            number_of_points = self.probes.shape[0]
            self.probes_rst = np.zeros((number_of_points, 3), dtype=np.double)
            self.el_owner = np.zeros((number_of_points), dtype=np.intc)
            self.glb_el_owner = np.zeros((number_of_points), dtype=np.intc)
            # self.rank_owner = np.zeros((number_of_points), dtype = np.intc)
            self.rank_owner = np.ones((number_of_points), dtype=np.intc) * -1000
            self.err_code = np.zeros(
                (number_of_points), dtype=np.intc
            )  # 0 not found, 1 is found
            self.test_pattern = np.ones(
                (number_of_points), dtype=np.double
            )  # test interpolation holder
        else:
            self.probes_rst = None
            self.el_owner = None
            self.glb_el_owner = None
            self.rank_owner = None
            self.err_code = None
            self.test_pattern = None

        # Now scatter the probes and all the codes according
        # to what is needed and each rank has a partition
        self.probe_partition_sendcount = probe_partition_sendcount
        self.probe_coord_partition_sendcount = probe_coord_partition_sendcount
        ## Double
        tmp = scatter_from_root(
            self.probes, probe_coord_partition_sendcount, io_rank, np.double, comm
        )
        self.probe_partition = tmp.reshape((int(tmp.size / 3), 3))
        tmp = scatter_from_root(
            self.probes_rst, probe_coord_partition_sendcount, io_rank, np.double, comm
        )
        self.probe_rst_partition = tmp.reshape((int(tmp.size / 3), 3))
        ## Int
        self.el_owner_partition = scatter_from_root(
            self.el_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.glb_el_owner_partition = scatter_from_root(
            self.glb_el_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.rank_owner_partition = scatter_from_root(
            self.rank_owner, probe_partition_sendcount, io_rank, np.intc, comm
        )
        self.err_code_partition = scatter_from_root(
            self.err_code, probe_partition_sendcount, io_rank, np.intc, comm
        )
        ## Double
        self.test_pattern_partition = scatter_from_root(
            self.test_pattern, probe_partition_sendcount, io_rank, np.double, comm
        )

        return

    def gather_probes_to_io_rank(self, io_rank, comm):
        """Gather the probes to the rank that is used to read them - rank0 by default"""

        root = io_rank
        sendbuf = self.probe_partition.reshape((self.probe_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if not isinstance(recvbuf, NoneType):
            self.probes[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.probe_rst_partition.reshape((self.probe_rst_partition.size))
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if not isinstance(recvbuf, NoneType):
            self.probes_rst[:, :] = recvbuf.reshape((int(recvbuf.size / 3), 3))[:, :]
        sendbuf = self.el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if not isinstance(recvbuf, NoneType):
            self.el_owner[:] = recvbuf[:]
        sendbuf = self.glb_el_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if not isinstance(recvbuf, NoneType):
            self.glb_el_owner[:] = recvbuf[:]
        sendbuf = self.rank_owner_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if not isinstance(recvbuf, NoneType):
            self.rank_owner[:] = recvbuf[:]
        sendbuf = self.err_code_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.intc, comm)
        if not isinstance(recvbuf, NoneType):
            self.err_code[:] = recvbuf[:]
        sendbuf = self.test_pattern_partition
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)
        if not isinstance(recvbuf, NoneType):
            self.test_pattern[:] = recvbuf[:]

        return

    def find_points(
        self,
        comm,
        find_points_comm_pattern="point_to_point",
        use_kdtree=True,
        test_tol=1e-4,
        elem_percent_expansion=0.01,
    ):
        """Public method to dins points across ranks and elements"""
        if comm.Get_rank() == 0:
            print(f"Using communication pattern: {find_points_comm_pattern}")

        if find_points_comm_pattern == "collective":
            self.find_points_collective_(
                comm,
                use_kdtree=use_kdtree,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
            )
        elif find_points_comm_pattern == "point_to_point":
            self.find_points_point_to_point_(
                comm,
                use_kdtree=use_kdtree,
                test_tol=test_tol,
                elem_percent_expansion=elem_percent_expansion,
            )

    def find_points_collective_(
        self, comm, use_kdtree=True, test_tol=1e-4, elem_percent_expansion=0.01
    ):
        """Find points using the collective implementation"""
        rank = comm.Get_rank()
        size = comm.Get_size()
        self.rank = rank

        # First each rank finds their bounding box
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree:
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = (
                get_bbox_centroids_and_max_dist(self.my_bbox)
            )

            # Build a KDtree with my information
            self.my_tree = KDTree(self.my_bbox_centroids)

        nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Now recursively make more ranks communicate to search for points that have not been found
        # Start checking if I have the local ones
        denom = 1
        j = 0
        while j < size and denom <= size + 1:

            # With this logic, first every rank checks by themselves and then 2
            # check together, then 4 ... up to when all of them check
            col = int(np.floor(((rank / denom))))
            if DEBUG:
                print(
                    f"rank: {rank}, finding points. start iteration: {j}. Color: {col}"
                )
            else:
                if rank == 0:
                    print(
                        f"rank: {rank}, finding points. start iteration: {j}. Color: {col}"
                    )
            start_time = MPI.Wtime()
            denom = denom * 2

            search_comm = comm.Split(color=col, key=rank)
            search_rank = search_comm.Get_rank()
            search_size = search_comm.Get_size()

            # Make each rank in the communicator broadcast their
            # bounding boxes to the others to search
            for broadcaster in range(0, search_size):

                # Points that I have not found to which rank or element belong to
                not_found = np.where(self.err_code_partition != 1)[0]
                n_not_found = not_found.size
                probe_not_found = self.probe_partition[not_found]
                probe_rst_not_found = self.probe_rst_partition[not_found]
                el_owner_not_found = self.el_owner_partition[not_found]
                glb_el_owner_not_found = self.glb_el_owner_partition[not_found]
                rank_owner_not_found = self.rank_owner_partition[not_found]
                err_code_not_found = self.err_code_partition[not_found]
                test_pattern_not_found = self.test_pattern_partition[not_found]

                # Tell every rank in the broadcaster of the local communicator their actual rank
                broadcaster_global_rank = np.ones((1), dtype=np.intc) * rank
                search_comm.Bcast(broadcaster_global_rank, root=broadcaster)

                # Tell every rank how much they need to allocate for the broadcaster bounding boxes
                nelv_in_broadcaster = np.ones((1), dtype=np.intc) * nelv
                search_comm.Bcast(nelv_in_broadcaster, root=broadcaster)

                # Allocate the recieve buffer for bounding boxes
                bbox_rec_buff = np.empty((nelv_in_broadcaster[0], 6), dtype=np.double)

                # Only in the broadcaster copy the data
                if search_rank == broadcaster:
                    bbox_rec_buff[:, :] = np.copy(self.my_bbox[:, :])

                # Broadcast the bounding boxes so each rank can check if the points are there
                search_comm.Bcast(bbox_rec_buff, root=broadcaster)

                # Only do the search, if my rank has not already searched the broadcaster
                if broadcaster_global_rank not in self.ranks_ive_checked:

                    if not use_kdtree:
                        # Find a candidate rank to check
                        i = 0
                        if self.progress_bar:
                            pbar = tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in range(0, bbox_rec_buff.shape[0]):
                                if pt_in_bbox(
                                    pt, bbox_rec_buff[e], rel_tol=elem_percent_expansion
                                ):
                                    found_candidate = True
                                    # el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    # rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()
                    elif use_kdtree:

                        # Get bbox centroids and max radius from center to
                        #  corner for the broadcaster
                        bbox_centroids, bbox_maxdist = get_bbox_centroids_and_max_dist(
                            bbox_rec_buff
                        )

                        # Create the KDtree
                        broadcaster_tree = KDTree(bbox_centroids)
                        # Query the tree with the probes to reduce the bbox search
                        candidate_elements = broadcaster_tree.query_ball_point(
                            x=probe_not_found,
                            r=bbox_maxdist,
                            p=2.0,
                            eps=elem_percent_expansion,
                            workers=1,
                            return_sorted=False,
                            return_length=False,
                        )

                        # Do a bbox search over the candidate elements, just as it used to be done
                        # (The KD tree allows to avoid searching ALL elements)
                        i = 0
                        if self.progress_bar:
                            pbar = tqdm(total=n_not_found)
                        for pt in probe_not_found:
                            found_candidate = False
                            for e in candidate_elements[i]:
                                if pt_in_bbox(
                                    pt, bbox_rec_buff[e], rel_tol=elem_percent_expansion
                                ):
                                    found_candidate = True
                                    # el_owner_not_found[i] = e
                                    err_code_not_found[i] = 1
                                    # rank_owner_not_found[i] = broadcaster_global_rank

                            if found_candidate:
                                i = i + 1
                                if self.progress_bar:
                                    pbar.update(1)
                                continue
                            i = i + 1
                            if self.progress_bar:
                                pbar.update(1)
                        if self.progress_bar:
                            pbar.close()

                    # Now let the brodcaster gather the points that the other ranks think it has
                    # broadcaster_is_candidate =
                    # np.where(rank_owner_not_found == broadcaster_global_rank)[0]
                    broadcaster_is_candidate = np.where(err_code_not_found == 1)[0]

                    self.ranks_ive_checked.append(broadcaster_global_rank[0])
                else:
                    # If this rank has already checked the broadcaster, just produce an empty list
                    # broadcaster_is_candidate = np.where(rank_owner_not_found == -10000)[0]
                    broadcaster_is_candidate = np.where(err_code_not_found == 10000)[0]

                probe_broadcaster_is_candidate = probe_not_found[
                    broadcaster_is_candidate
                ]
                probe_rst_broadcaster_is_candidate = probe_rst_not_found[
                    broadcaster_is_candidate
                ]
                el_owner_broadcaster_is_candidate = el_owner_not_found[
                    broadcaster_is_candidate
                ]
                glb_el_owner_broadcaster_is_candidate = glb_el_owner_not_found[
                    broadcaster_is_candidate
                ]
                rank_owner_broadcaster_is_candidate = rank_owner_not_found[
                    broadcaster_is_candidate
                ]
                err_code_broadcaster_is_candidate = err_code_not_found[
                    broadcaster_is_candidate
                ]
                test_pattern_broadcaster_is_candidate = test_pattern_not_found[
                    broadcaster_is_candidate
                ]

                root = broadcaster
                tmp, probe_sendcount_broadcaster_is_candidate = gather_in_root(
                    probe_broadcaster_is_candidate.reshape(
                        (probe_broadcaster_is_candidate.size)
                    ),
                    root,
                    np.double,
                    search_comm,
                )
                if not isinstance(tmp, NoneType):
                    probe_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))

                # For debugging
                if broadcaster_global_rank == TR and rank == TR and DEBUG is True:
                    print(probe_sendcount_broadcaster_is_candidate / 3)
                    # print(probe_broadcaster_has)

                tmp, _ = gather_in_root(
                    probe_rst_broadcaster_is_candidate.reshape(
                        (probe_rst_broadcaster_is_candidate.size)
                    ),
                    root,
                    np.double,
                    search_comm,
                )
                if not isinstance(tmp, NoneType):
                    probe_rst_broadcaster_has = tmp.reshape((int(tmp.size / 3), 3))
                (
                    el_owner_broadcaster_has,
                    el_owner_sendcount_broadcaster_is_candidate,
                ) = gather_in_root(
                    el_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                glb_el_owner_broadcaster_has, _ = gather_in_root(
                    glb_el_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                rank_owner_broadcaster_has, _ = gather_in_root(
                    rank_owner_broadcaster_is_candidate, root, np.intc, search_comm
                )
                err_code_broadcaster_has, _ = gather_in_root(
                    err_code_broadcaster_is_candidate, root, np.intc, search_comm
                )
                test_pattern_broadcaster_has, _ = gather_in_root(
                    test_pattern_broadcaster_is_candidate, root, np.double, search_comm
                )

                # Now let the broadcaster check if it really had the point.
                # It will go through all the elements again and check rst coordinates
                if search_rank == broadcaster:

                    probes_info = {}
                    probes_info["probes"] = probe_broadcaster_has
                    probes_info["probes_rst"] = probe_rst_broadcaster_has
                    probes_info["el_owner"] = el_owner_broadcaster_has
                    probes_info["glb_el_owner"] = glb_el_owner_broadcaster_has
                    probes_info["rank_owner"] = rank_owner_broadcaster_has
                    probes_info["err_code"] = err_code_broadcaster_has
                    probes_info["test_pattern"] = test_pattern_broadcaster_has
                    probes_info["rank"] = broadcaster_global_rank
                    probes_info["offset_el"] = self.offset_el

                    mesh_info = {}
                    mesh_info["x"] = self.x
                    mesh_info["y"] = self.y
                    mesh_info["z"] = self.z
                    mesh_info["bbox"] = self.my_bbox
                    if hasattr(self, "my_tree"):
                        mesh_info["kd_tree"] = self.my_tree
                        mesh_info["bbox_max_dist"] = self.my_bbox_maxdist

                    settings = {}
                    settings["not_found_code"] = -10
                    settings["use_test_pattern"] = True
                    settings["elem_percent_expansion"] = elem_percent_expansion
                    settings["progress_bar"] = self.progress_bar

                    buffers = {}
                    buffers["r"] = self.r
                    buffers["s"] = self.s
                    buffers["t"] = self.t
                    buffers["test_interp"] = self.test_interp

                    [
                        probe_broadcaster_has,
                        probe_rst_broadcaster_has,
                        el_owner_broadcaster_has,
                        glb_el_owner_broadcaster_has,
                        rank_owner_broadcaster_has,
                        err_code_broadcaster_has,
                        test_pattern_broadcaster_has,
                    ] = self.ei.find_rst(
                        probes_info, mesh_info, settings, buffers=buffers
                    )

                # Now scatter the results back to all the other ranks
                root = broadcaster
                if search_rank == root:
                    sendbuf = probe_broadcaster_has.reshape(probe_broadcaster_has.size)
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
                )
                probe_broadcaster_is_candidate[:, :] = recvbuf.reshape(
                    (int(recvbuf.size / 3), 3)
                )[:, :]
                if search_rank == root:
                    sendbuf = probe_rst_broadcaster_has.reshape(
                        probe_rst_broadcaster_has.size
                    )
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    probe_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
                )
                probe_rst_broadcaster_is_candidate[:, :] = recvbuf.reshape(
                    (int(recvbuf.size / 3), 3)
                )[:, :]
                if search_rank == root:
                    sendbuf = el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    # sendbuf = el_owner_broadcaster_has+self.offset_el
                    sendbuf = glb_el_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                glb_el_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = rank_owner_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                rank_owner_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = err_code_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.intc,
                    search_comm,
                )
                err_code_broadcaster_is_candidate[:] = recvbuf[:]
                if search_rank == root:
                    sendbuf = test_pattern_broadcaster_has
                else:
                    sendbuf = None
                recvbuf = scatter_from_root(
                    sendbuf,
                    el_owner_sendcount_broadcaster_is_candidate,
                    root,
                    np.double,
                    search_comm,
                )
                test_pattern_broadcaster_is_candidate[:] = recvbuf[:]

                # Now that the data is back at the original ranks,
                # put it in the place of the not found list that it should be
                probe_not_found[broadcaster_is_candidate, :] = (
                    probe_broadcaster_is_candidate[:, :]
                )
                probe_rst_not_found[broadcaster_is_candidate, :] = (
                    probe_rst_broadcaster_is_candidate[:, :]
                )
                el_owner_not_found[broadcaster_is_candidate] = (
                    el_owner_broadcaster_is_candidate[:]
                )
                glb_el_owner_not_found[broadcaster_is_candidate] = (
                    glb_el_owner_broadcaster_is_candidate[:]
                )
                rank_owner_not_found[broadcaster_is_candidate] = (
                    rank_owner_broadcaster_is_candidate[:]
                )
                err_code_not_found[broadcaster_is_candidate] = (
                    err_code_broadcaster_is_candidate[:]
                )
                test_pattern_not_found[broadcaster_is_candidate] = (
                    test_pattern_broadcaster_is_candidate[:]
                )

                # at the end of the broadcaster run, update the information
                #  from the previously not found data
                self.probe_partition[not_found, :] = probe_not_found[:, :]
                self.probe_rst_partition[not_found] = probe_rst_not_found[:, :]
                self.el_owner_partition[not_found] = el_owner_not_found[:]
                self.glb_el_owner_partition[not_found] = glb_el_owner_not_found[:]
                self.rank_owner_partition[not_found] = rank_owner_not_found[:]
                self.err_code_partition[not_found] = err_code_not_found[:]
                self.test_pattern_partition[not_found] = test_pattern_not_found[:]

            if DEBUG:
                print(
                    f"rank: {rank}, finding points. finished iteration: {j}. time(s): {MPI.Wtime() - start_time}"
                )
            else:
                if rank == 0:
                    print(
                        f"rank: {rank}, finding points. finished iteration: {j}. time(s): {MPI.Wtime() - start_time}"
                    )

            j = j + 1
            search_comm.Free()

        # Final check
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            # were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

            # Check also if the rst are too big, then it needs to be outside
            # if ( abs(self.probe_rst_partition[j, 0]) +  abs(self.probe_rst_partition[j, 1]) +
            # abs(self.probe_rst_partition[j, 2]) ) > 3.5:
            #    self.err_code_partition[j] = 0

        return

    def find_points_point_to_point_(
        self, comm, use_kdtree=True, test_tol=1e-4, elem_percent_expansion=0.01
    ):
        """Find points using the point to point implementation"""
        rank = comm.Get_rank()
        self.rank = rank

        if DEBUG:
            print(f"rank: {rank}, finding points. start")
        else:
            if rank == 0:
                print(f"rank: {rank}, finding points. start")
        start_time = MPI.Wtime()

        # First each rank finds their bounding box
        self.my_bbox = get_bbox_from_coordinates(self.x, self.y, self.z)

        if use_kdtree:
            # Get bbox centroids and max radius from center to corner
            self.my_bbox_centroids, self.my_bbox_maxdist = (
                get_bbox_centroids_and_max_dist(self.my_bbox)
            )

            # Build a KDtree with my information
            self.my_tree = KDTree(self.my_bbox_centroids)

        # nelv = self.x.shape[0]
        self.ranks_ive_checked = []

        # Get candidate ranks from a global kd tree
        # These are the destination ranks
        my_dest = get_candidate_ranks(self, comm)

        # Get a global array with the candidates in all other
        # ranks to determine the best way to communicate
        global_rank_candidate = get_global_candidate_ranks(self, comm, my_dest)

        # Get the ranks that have me in their dest, so they become
        # my sources (This tank will check their data)
        my_source = np.where(np.any(global_rank_candidate == rank, axis=1))[0]

        # Create temporary arrays that store the points that have not been found
        not_found = np.where(self.err_code_partition != 1)[0]
        n_not_found = not_found.size
        probe_not_found = self.probe_partition[not_found]
        probe_rst_not_found = self.probe_rst_partition[not_found]
        el_owner_not_found = self.el_owner_partition[not_found]
        glb_el_owner_not_found = self.glb_el_owner_partition[not_found]
        rank_owner_not_found = self.rank_owner_partition[not_found]
        err_code_not_found = self.err_code_partition[not_found]
        test_pattern_not_found = self.test_pattern_partition[not_found]

        # Tell every rank how many points not found each other rank has
        not_found_in_this_rank = np.ones((1), dtype=np.intc) * n_not_found
        not_found_in_all_ranks = np.zeros((comm.Get_size()), dtype=np.intc)
        comm.Allgather(
            [not_found_in_this_rank, MPI.INT], [not_found_in_all_ranks, MPI.INT]
        )  # This allgather can be changed with point2point

        # Check how many buffers to create to recieve points
        # from other ranks that think this rank is a candidate
        n_buff = len(my_source)
        # Create buffer for data from other ranks
        buff_probes = []
        buff_probes_rst = []
        buff_el_owner = []
        buff_glb_el_owner = []
        buff_rank_owner = []
        buff_err_code = []
        buff_test_pattern = []
        for ni in range(n_buff):
            # The points in each buffer depend on the points on the sending rank
            npt = not_found_in_all_ranks[my_source[ni]]
            buff_probes.append(np.zeros((npt, 3), dtype=np.double))
            buff_probes_rst.append(np.zeros((npt, 3), dtype=np.double))
            buff_el_owner.append(np.zeros((npt), dtype=np.intc))
            buff_glb_el_owner.append(np.zeros((npt), dtype=np.intc))
            buff_rank_owner.append(np.ones((npt), dtype=np.intc) * -1000)
            buff_err_code.append(
                np.zeros((npt), dtype=np.intc)
            )  # 0 not found, 1 is found
            buff_test_pattern.append(
                np.ones((npt), dtype=np.double)
            )  # test interpolation holder

        # This rank will send its data to the other ranks in the dest list.
        # These buffers will be used to store the data when the points are sent back
        on_buff = len(my_dest)
        obuff_probes = []
        obuff_probes_rst = []
        obuff_el_owner = []
        obuff_glb_el_owner = []
        obuff_rank_owner = []
        obuff_err_code = []
        obuff_test_pattern = []
        for ni in range(on_buff):
            # The points in each buffer are the same points in this rank
            npt = n_not_found
            obuff_probes.append(np.zeros((npt, 3), dtype=np.double))
            obuff_probes_rst.append(np.zeros((npt, 3), dtype=np.double))
            obuff_el_owner.append(np.zeros((npt), dtype=np.intc))
            obuff_glb_el_owner.append(np.zeros((npt), dtype=np.intc))
            obuff_rank_owner.append(np.ones((npt), dtype=np.intc) * -1000)
            obuff_err_code.append(
                np.zeros((npt), dtype=np.intc)
            )  # 0 not found, 1 is found
            obuff_test_pattern.append(
                np.ones((npt), dtype=np.double)
            )  # test interpolation holder

        # Set the request to Recieve the data from the other ranks that have me as a candidate
        req_list_r = []
        for source_index in range(0, len(my_source)):
            source = my_source[source_index]
            req_list_r.append([])
            req_list_r[source_index].append(
                comm.Irecv(buff_probes[source_index], source=source, tag=1)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_probes_rst[source_index], source=source, tag=2)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_el_owner[source_index], source=source, tag=3)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_glb_el_owner[source_index], source=source, tag=4)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_rank_owner[source_index], source=source, tag=5)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_err_code[source_index], source=source, tag=6)
            )
            req_list_r[source_index].append(
                comm.Irecv(buff_test_pattern[source_index], source=source, tag=7)
            )

        # Set and complete the request to send my points to my candidates
        req_list_s = []
        dest_index = -1
        for dest in my_dest:
            dest_index = dest_index + 1
            req_list_s.append([])
            req_list_s[dest_index].append(comm.Isend(probe_not_found, dest=dest, tag=1))
            req_list_s[dest_index].append(
                comm.Isend(probe_rst_not_found, dest=dest, tag=2)
            )
            req_list_s[dest_index].append(
                comm.Isend(el_owner_not_found, dest=dest, tag=3)
            )
            req_list_s[dest_index].append(
                comm.Isend(glb_el_owner_not_found, dest=dest, tag=4)
            )
            req_list_s[dest_index].append(
                comm.Isend(rank_owner_not_found, dest=dest, tag=5)
            )
            req_list_s[dest_index].append(
                comm.Isend(err_code_not_found, dest=dest, tag=6)
            )
            req_list_s[dest_index].append(
                comm.Isend(test_pattern_not_found, dest=dest, tag=7)
            )

            # complete the send request
            for req in req_list_s[dest_index]:
                req.wait()

        # Complete the request to recieve the data in the buffers
        for source_index in range(0, len(my_source)):
            for req in req_list_r[source_index]:
                req.wait()

        # Set the information for the coordinate search in this rank
        mesh_info = {}
        mesh_info["x"] = self.x
        mesh_info["y"] = self.y
        mesh_info["z"] = self.z
        mesh_info["bbox"] = self.my_bbox
        if hasattr(self, "my_tree"):
            mesh_info["kd_tree"] = self.my_tree
            mesh_info["bbox_max_dist"] = self.my_bbox_maxdist

        settings = {}
        settings["not_found_code"] = -10
        settings["use_test_pattern"] = True
        settings["elem_percent_expansion"] = elem_percent_expansion
        settings["progress_bar"] = self.progress_bar

        buffers = {}
        buffers["r"] = self.r
        buffers["s"] = self.s
        buffers["t"] = self.t
        buffers["test_interp"] = self.test_interp

        # Now find the rst coordinates for the points stored in each of the buffers
        for source_index in range(0, len(my_source)):

            probes_info = {}
            probes_info["probes"] = buff_probes[source_index]
            probes_info["probes_rst"] = buff_probes_rst[source_index]
            probes_info["el_owner"] = buff_el_owner[source_index]
            probes_info["glb_el_owner"] = buff_glb_el_owner[source_index]
            probes_info["rank_owner"] = buff_rank_owner[source_index]
            probes_info["err_code"] = buff_err_code[source_index]
            probes_info["test_pattern"] = buff_test_pattern[source_index]
            probes_info["rank"] = rank
            probes_info["offset_el"] = self.offset_el

            [
                buff_probes[source_index],
                buff_probes_rst[source_index],
                buff_el_owner[source_index],
                buff_glb_el_owner[source_index],
                buff_rank_owner[source_index],
                buff_err_code[source_index],
                buff_test_pattern[source_index],
            ] = self.ei.find_rst(probes_info, mesh_info, settings, buffers=buffers)

        # Set the request to Recieve back the data that I have sent to my candidates
        oreq_list_r = []
        for source_index in range(0, len(my_dest)):
            source = my_dest[source_index]
            oreq_list_r.append([])
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_probes[source_index], source=source, tag=11)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_probes_rst[source_index], source=source, tag=12)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_el_owner[source_index], source=source, tag=13)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_glb_el_owner[source_index], source=source, tag=14)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_rank_owner[source_index], source=source, tag=15)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_err_code[source_index], source=source, tag=16)
            )
            oreq_list_r[source_index].append(
                comm.Irecv(obuff_test_pattern[source_index], source=source, tag=17)
            )

        # Set the request to send this data back to the rank that sent it to me
        oreq_list_s = []
        for dest_index in range(0, len(my_source)):
            dest = my_source[dest_index]
            oreq_list_s.append([])
            oreq_list_s[dest_index].append(
                comm.Isend(buff_probes[dest_index], dest=dest, tag=11)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_probes_rst[dest_index], dest=dest, tag=12)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_el_owner[dest_index], dest=dest, tag=13)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_glb_el_owner[dest_index], dest=dest, tag=14)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_rank_owner[dest_index], dest=dest, tag=15)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_err_code[dest_index], dest=dest, tag=16)
            )
            oreq_list_s[dest_index].append(
                comm.Isend(buff_test_pattern[dest_index], dest=dest, tag=17)
            )

            # Complete the send request
            for req in oreq_list_s[dest_index]:
                req.wait()

        # Complete the request to recieve the data in the buffers
        for index in range(0, len(my_dest)):
            for req in oreq_list_r[index]:
                req.wait()

        # Now loop trhough all the points in the buffers that
        # have been sent back and determine which point was found
        for point in range(0, n_not_found):

            # These are the error code and test patterns for
            # this point from all the ranks that sent back
            all_err_codes = [arr[point] for arr in obuff_err_code]
            all_test_patterns = [arr[point] for arr in obuff_test_pattern]

            # Check if any rank had certainty that it had found the point
            found_err_code = np.where(np.array(all_err_codes) == 1)[0]

            # If the point was found in any rank, just choose the first
            # one in the list (in case there was more than one founder):
            if found_err_code.size > 0:
                index = found_err_code[0]
                self.probe_partition[point, :] = obuff_probes[index][point, :]
                self.probe_rst_partition[point, :] = obuff_probes_rst[index][point, :]
                self.el_owner_partition[point] = obuff_el_owner[index][point]
                self.glb_el_owner_partition[point] = obuff_glb_el_owner[index][point]
                self.rank_owner_partition[point] = obuff_rank_owner[index][point]
                self.err_code_partition[point] = obuff_err_code[index][point]
                self.test_pattern_partition[point] = obuff_test_pattern[index][point]

                # skip the rest of the loop
                continue

            # If the point was not found with certainty, then choose as
            # owner the the one that produced the smaller error in the test pattern
            min_test_pattern = np.where(
                np.array(all_test_patterns) == np.array(all_test_patterns).min()
            )[0]
            if min_test_pattern.size > 0:
                index = min_test_pattern[0]
                self.probe_partition[point, :] = obuff_probes[index][point, :]
                self.probe_rst_partition[point, :] = obuff_probes_rst[index][point, :]
                self.el_owner_partition[point] = obuff_el_owner[index][point]
                self.glb_el_owner_partition[point] = obuff_glb_el_owner[index][point]
                self.rank_owner_partition[point] = obuff_rank_owner[index][point]
                self.err_code_partition[point] = obuff_err_code[index][point]
                self.test_pattern_partition[point] = obuff_test_pattern[index][point]

        # Go through the points again, if the test pattern was
        #  too large, mark that point as not found
        for j in range(0, len(self.test_pattern_partition)):
            # After all iteration are done, check if some points
            #  were not found. Use the error code and the test pattern
            if (
                self.err_code_partition[j] != 1
                and self.test_pattern_partition[j] > test_tol
            ):
                self.err_code_partition[j] = 0

        if DEBUG:
            print(
                f"rank: {rank}, finding points. finished. time(s): {MPI.Wtime() - start_time}"
            )
        else:
            if rank == 0:
                print(
                    f"rank: {rank}, finding points. finished. time(s): {MPI.Wtime() - start_time}"
                )

        return

    def redistribute_probes_to_owners_from_io_rank(self, io_rank, comm):
        """Redistribute the probes to the ranks that
        have been determined in the search"""

        rank = comm.Get_rank()
        size = comm.Get_size()

        probes = self.probes
        probes_rst = self.probes_rst
        el_owner = self.el_owner
        rank_owner = self.rank_owner
        err_code = self.err_code

        # Sort the points by rank to scatter them easily
        if rank == io_rank:

            # Before the sorting, assign all the not found probes to rank zero.
            # The points with error code 0 will be ignored in the interpolation routine
            # Doing this will avoid an error when after interpolating we try to gather the points
            self.rank_owner[np.where(err_code == 0)] = 0
            rank_owner[np.where(err_code == 0)] = 0

            sort_by_rank = np.argsort(rank_owner)

            sorted_probes = probes[sort_by_rank]
            sorted_probes_rst = probes_rst[sort_by_rank]
            sorted_el_owner = el_owner[sort_by_rank]
            sorted_rank_owner = rank_owner[sort_by_rank]
            sorted_err_code = err_code[sort_by_rank]
        else:
            sort_by_rank = None

        # Check the sendcounts in number of probes
        sendcounts = np.zeros((size), dtype=np.intc)
        if rank == io_rank:
            for root in range(0, size):
                sendcounts[root] = len(np.where(rank_owner == root)[0])
        comm.Bcast(sendcounts, root=0)

        root = io_rank
        # Redistribute probes
        if rank == root:
            sendbuf = sorted_probes.reshape((sorted_probes.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts * 3, root, np.double, comm)
        my_probes = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute probes rst
        if rank == root:
            sendbuf = sorted_probes_rst.reshape((sorted_probes_rst.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts * 3, root, np.double, comm)
        my_probes_rst = recvbuf.reshape((int(recvbuf.size / 3), 3))

        # Redistribute err_code
        if rank == root:
            sendbuf = sorted_err_code.reshape((sorted_err_code.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_err_code = recvbuf

        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_el_owner.reshape((sorted_el_owner.size))
            # print(sendbuf)
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        # print(recvbuf)
        my_el_owner = recvbuf

        # Redistribute el_owner
        if rank == root:
            sendbuf = sorted_rank_owner.reshape((sorted_rank_owner.size))
        else:
            sendbuf = None
        recvbuf = scatter_from_root(sendbuf, sendcounts, root, np.intc, comm)
        my_rank_owner = recvbuf

        self.my_probes = my_probes
        self.my_probes_rst = my_probes_rst
        self.my_err_code = my_err_code
        self.my_el_owner = my_el_owner
        self.my_rank_owner = my_rank_owner
        self.sendcounts = sendcounts
        self.sort_by_rank = sort_by_rank

        return

    def interpolate_field_from_rst(self, sampled_field):
        """Interpolate the field from the rst coordinates found"""

        # Probes info
        probes_info = {}
        probes_info["probes"] = self.my_probes
        probes_info["probes_rst"] = self.my_probes_rst
        probes_info["el_owner"] = self.my_el_owner
        probes_info["err_code"] = self.my_err_code

        # Settings
        settings = {}
        settings["progress_bar"] = self.progress_bar

        sampled_field_at_probe = self.ei.interpolate_field_from_rst(
            probes_info,
            interpolation_buffer=self.test_interp,
            sampled_field=sampled_field,
            settings=settings,
        )

        return sampled_field_at_probe


def pt_in_bbox(pt, bbox, rel_tol=0.01):
    """Determine if the point is in the bounding box"""
    # rel_tol=1% enlargement of the bounding box by default

    state = False
    found_x = False
    found_y = False
    found_z = False

    d = bbox[1] - bbox[0]
    tol = d * rel_tol / 2
    if pt[0] >= bbox[0] - tol and pt[0] <= bbox[1] + tol:
        found_x = True

    d = bbox[3] - bbox[2]
    tol = d * rel_tol / 2
    if pt[1] >= bbox[2] - tol and pt[1] <= bbox[3] + tol:
        found_y = True

    d = bbox[5] - bbox[4]
    tol = d * rel_tol / 2
    if pt[2] >= bbox[4] - tol and pt[2] <= bbox[5] + tol:
        found_z = True

    if found_x is True and found_y == True is found_z == True:
        state = True
    else:
        state = False

    return state


def get_bbox_from_coordinates(x, y, z):

    nelv = x.shape[0]
    # lx = x.shape[3]  # This is not a mistake. This is how the data is read
    # ly = x.shape[2]
    # lz = x.shape[1]

    bbox = np.zeros((nelv, 6), dtype=np.double)

    for e in range(0, nelv):
        bbox[e, 0] = np.min(x[e, :, :, :])
        bbox[e, 1] = np.max(x[e, :, :, :])
        bbox[e, 2] = np.min(y[e, :, :, :])
        bbox[e, 3] = np.max(y[e, :, :, :])
        bbox[e, 4] = np.min(z[e, :, :, :])
        bbox[e, 5] = np.max(z[e, :, :, :])

    return bbox


def get_bbox_centroids_and_max_dist(bbox):

    # Then find the centroids of each bbox and the maximun bbox radious from centroid to corner
    bbox_dist = np.zeros((bbox.shape[0], 3))
    bbox_dist[:, 0] = bbox[:, 1] - bbox[:, 0]
    bbox_dist[:, 1] = bbox[:, 3] - bbox[:, 2]
    bbox_dist[:, 2] = bbox[:, 5] - bbox[:, 4]

    bbox_max_dist = np.max(
        np.sqrt(bbox_dist[:, 0] ** 2 + bbox_dist[:, 1] ** 2 + bbox_dist[:, 2] ** 2) / 2
    )

    bbox_centroid = np.zeros((bbox.shape[0], 3))
    bbox_centroid[:, 0] = bbox[:, 0] + bbox_dist[:, 0] / 2
    bbox_centroid[:, 1] = bbox[:, 2] + bbox_dist[:, 1] / 2
    bbox_centroid[:, 2] = bbox[:, 4] + bbox_dist[:, 2] / 2

    return bbox_centroid, bbox_max_dist


def get_communication_pairs(self, global_rank_candidate_dict, comm):
    """Get the pairs of ranks that will communicate"""
    size = comm.Get_size()

    # Create a list with all the ranks
    ranks = [ii for ii in range(0, size)]

    # Get all unique pairs and colours
    if isinstance(global_rank_candidate_dict, NoneType):
        pairs = list(combinations(ranks, 2))
    else:
        pairs_temp = [
            (i, j) for i in range(0, size) for j in global_rank_candidate_dict[i]
        ]
        pairs = []
        for tt in range(0, len(pairs_temp)):
            i = pairs_temp[tt][0]
            j = pairs_temp[tt][1]
            if (i, j) not in pairs and (j, i) not in pairs and i != j:
                pairs.append((i, j))

    # Find the pairs in which my rank is part of
    rank = comm.Get_rank()
    my_pairs = [(i, j) for i, j in pairs if i == rank or j == rank]
    my_source_dest = [i if i != rank else j for (i, j) in my_pairs]
    my_source_dest.append(rank)
    my_source_dest.sort()

    return my_pairs, my_source_dest


def get_candidate_ranks(self, comm):
    """Get candiate ranks for each rank separately"""

    size = comm.Get_size()

    # Find the bounding box of the rank to create a global but "sparse" kdtree
    rank_bbox = np.zeros((1, 6), dtype=np.double)
    rank_bbox[0, 0] = np.min(self.x)
    rank_bbox[0, 1] = np.max(self.x)
    rank_bbox[0, 2] = np.min(self.y)
    rank_bbox[0, 3] = np.max(self.y)
    rank_bbox[0, 4] = np.min(self.z)
    rank_bbox[0, 5] = np.max(self.z)

    rank_bbox_dist = np.zeros((1, 3), dtype=np.double)
    rank_bbox_dist[0, 0] = rank_bbox[0, 1] - rank_bbox[0, 0]
    rank_bbox_dist[0, 1] = rank_bbox[0, 3] - rank_bbox[0, 2]
    rank_bbox_dist[0, 2] = rank_bbox[0, 5] - rank_bbox[0, 4]
    rank_bbox_max_dist = np.max(
        np.sqrt(
            rank_bbox_dist[:, 0] ** 2
            + rank_bbox_dist[:, 1] ** 2
            + rank_bbox_dist[:, 2] ** 2
        )
        / 2
    )

    rank_bbox_centroid = np.zeros((1, 3))
    rank_bbox_centroid[:, 0] = rank_bbox[:, 0] + rank_bbox_dist[:, 0] / 2
    rank_bbox_centroid[:, 1] = rank_bbox[:, 2] + rank_bbox_dist[:, 1] / 2
    rank_bbox_centroid[:, 2] = rank_bbox[:, 4] + rank_bbox_dist[:, 2] / 2

    global_centroids = np.zeros((size * 3), dtype=np.double)
    global_max_dist = np.zeros((size), dtype=np.double)

    # Gather in all ranks
    comm.Allgather(
        [rank_bbox_centroid.flatten(), MPI.DOUBLE], [global_centroids, MPI.DOUBLE]
    )
    comm.Allgather([rank_bbox_max_dist, MPI.DOUBLE], [global_max_dist, MPI.DOUBLE])
    global_centroids = global_centroids.reshape((size, 3))
    # Create a tree with the rank centroids
    self.global_tree = KDTree(global_centroids)
    candidate_ranks_per_point = self.global_tree.query_ball_point(
        x=self.probe_partition,
        r=np.max(global_max_dist),
        p=2.0,
        eps=1e-8,
        workers=1,
        return_sorted=False,
        return_length=False,
    )

    flattened_list = [item for sublist in candidate_ranks_per_point for item in sublist]
    candidate_ranks = list(set(flattened_list))

    return candidate_ranks


def get_global_candidate_ranks(self, comm, candidate_ranks):
    """Get an array with the candidate ranks of all ranks"""

    size = comm.Get_size()

    # Get arrays with number of candidates per rank
    ## Tell how many there are per rank
    n_candidate_ranks_per_rank = np.zeros((size), dtype=np.intc)
    sendbuf = np.ones((1), dtype=np.intc) * len(candidate_ranks)
    comm.Allgather([sendbuf, MPI.INT], [n_candidate_ranks_per_rank, MPI.INT])

    ## Allocate an array in all ranks that tells which are the rank candidates in all other ranks
    nc = np.max(n_candidate_ranks_per_rank)
    rank_candidates_in_all_ranks = np.zeros((size, nc), dtype=np.intc)
    rank_candidates_in_this_rank = (
        np.ones((1, nc), dtype=np.intc) * -1
    )  # Set default to -1 to filter out easily later
    for i in range(0, len(candidate_ranks)):
        rank_candidates_in_this_rank[0, i] = candidate_ranks[i]
    comm.Allgather(
        [rank_candidates_in_this_rank, MPI.INT], [rank_candidates_in_all_ranks, MPI.INT]
    )  # This all gather can be changed for gather and broadcast

    ## Filter out the -1 entries for ranks
    # that had less candidates than the maximun determined before
    # global_rank_candidate_dict = {i: [] for i in range(size)}
    # for i in range(0, size):
    #    for j in range(0, nc):
    #        if rank_candidates_in_all_ranks[i,j] != -1:
    #            global_rank_candidate_dict[i].append(rank_candidates_in_all_ranks[i,j])
    ## Delete the big array
    # del rank_candidates_in_all_ranks

    return rank_candidates_in_all_ranks
