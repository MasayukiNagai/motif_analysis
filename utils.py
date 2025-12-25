import subprocess
import h5py
import numpy as np


# ============================================================================
# Run TF-MoDISco-lite
# ============================================================================
def run_tfmodisco(
    seq_path,
    attr_path,
    output_h5="tfmodisco_results.h5",
    output_dir="tfmodisco_report",
    n_seqlets=2000,
    window_size=1000,
):
    """Run tfmodisco-lite for motif discovery"""

    # Run modisco motifs
    cmd_motifs = [
        "modisco",
        "motifs",
        "-s",
        str(seq_path),
        "-a",
        str(attr_path),
        "-n",
        str(n_seqlets),
        "-o",
        str(output_h5),
        "-w",
        str(window_size),
    ]
    print("CMD:", " ".join(cmd_motifs))
    subprocess.run(cmd_motifs, check=True)

    # Generate report
    cmd_report = ["modisco", "report", "-i", str(output_h5), "-o", str(output_dir)]
    print("CMD:", " ".join(cmd_report))
    subprocess.run(cmd_report, check=True)

    print(f"TF-MoDISco results saved to {output_h5}")
    print(f"Report saved to {output_dir}")


# ============================================================================
# Get moifs from TF-MoDISco results (& save them in the MEME format)
# ============================================================================
def get_motifs_from_tfmodisco(
    modisco_h5_path,
    use_cwm=False,
    trim_threshold=0.0,
    output_meme_path=None,
):
    """
    Extract motifs from a TF-MoDISco HDF5 file as numpy arrays.

    Motifs are returned as an OrderedDict mapping motif names to numpy arrays.
    The array represents:
      - PFM (position frequency matrix) if use_cwm=False
      - CWM (contribution weight matrix) if use_cwm=True

    Motifs are sorted first by metacluster (pos_patterns, then neg_patterns),
    and then by numeric pattern index within each metacluster.

    Flanking positions can optionally be trimmed using TF-MoDISco-lite semantics:
    positions are kept if the maximum base probability at that position is
    >= trim_threshold.

    If output_meme_path is provided, motifs are also written in MEME format.

    Parameters
    ----------
    modisco_h5_path : str or Path
        Path to the TF-MoDISco output HDF5 file.
    use_cwm : bool, default False
        If True, return CWMs (contrib_scores).
        If False, return PFMs (sequence).
    trim_threshold : float, default 0.0
        Trim flanks until a position has at least one base
        with probability >= trim_threshold.
    output_meme_path : str or Path, optional
        If provided, write motifs to this path in MEME format.

    Returns
    -------
    OrderedDict[str, np.ndarray]
        Mapping from motif name to (L, 4) PFM or CWM array.
    """
    import re
    import h5py
    import numpy as np
    from collections import OrderedDict

    def extract_pattern_number(pattern_name):
        match = re.search(r"pattern_(\d+)", pattern_name)
        return int(match.group(1)) if match else 0

    def trim_motif(matrix, threshold):
        prob_matrix = np.abs(matrix) + 1e-6
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        max_probs = prob_matrix.max(axis=1)
        keep = max_probs >= threshold

        if not keep.any():
            return matrix

        idx = np.where(keep)[0]
        return matrix[idx[0] : idx[-1] + 1]

    def to_prob_matrix(matrix):
        pm = np.abs(matrix) + 1e-6
        return pm / pm.sum(axis=1, keepdims=True)

    with h5py.File(modisco_h5_path, "r") as f:
        all_motifs = []

        for metacluster in ["pos_patterns", "neg_patterns"]:
            if metacluster not in f:
                continue

            for pattern_name in f[metacluster].keys():
                pattern = f[metacluster][pattern_name]
                motif_name = f"{metacluster}_{pattern_name}"

                value = None
                if use_cwm and "contrib_scores" in pattern:
                    value = pattern["contrib_scores"][:]
                elif not use_cwm and "sequence" in pattern:
                    value = pattern["sequence"][:]

                if value is not None:
                    value = trim_motif(value, trim_threshold)

                all_motifs.append(
                    {
                        "name": motif_name,
                        "metacluster": metacluster,
                        "pattern_num": extract_pattern_number(pattern_name),
                        "value": value,
                    }
                )

        all_motifs.sort(
            key=lambda x: (
                0 if x["metacluster"] == "pos_patterns" else 1,
                x["pattern_num"],
            )
        )

        motif_data = OrderedDict(
            (m["name"], m["value"].transpose(1, 0)) for m in all_motifs
        )

    if output_meme_path is not None:
        with open(output_meme_path, "w") as f:
            f.write("MEME version 4\n\n")
            f.write("ALPHABET= ACGT\n\n")
            f.write("strands: + -\n\n")
            f.write("Background letter frequencies\n")
            f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

            for name, mat in motif_data.items():
                prob = to_prob_matrix(mat)
                L = prob.shape[0]

                f.write(f"MOTIF {name}\n")
                f.write(f"letter-probability matrix: alength= 4 w= {L}\n")
                for row in prob:
                    f.write(" ".join(f"{x:.6f}" for x in row) + "\n")
                f.write("\n")

    return motif_data


# ============================================================================
# Scan sequences with memelite FIMO
# ============================================================================
# def scan_sequences_with_fimo(motif_dict, X):
#     """Scan sequences with memelite FIMO using given motifs"""
#     from memelite.fimo import fimo
#     hits = fimo(
#         motifs=motif_dict,
#         sequences=X,
#     )
#     return hits


# ============================================================================
# Build sequence × motif matrix from memelite FIMO results
# ============================================================================
def build_motif_matrix(fimo_df_list, n_sequences, attributions=None, aggregate="sum"):
    """
    Build sequence x motif matrix from memelite FIMO results.

    Parameters
    ----------
    fimo_df_list : list of pandas.DataFrame
        List of DataFrames from memelite FIMO (one per motif).
    n_sequences : int
        Total number of sequences.
    attributions : array_like, optional
        Attribution array with shape (N, 4, L) for weighting hits. Default is None.
    aggregate : {'max', 'sum', 'count', 'binary'}, optional
        Aggregation method for multiple hits. Default is 'max'.

    Returns
    -------
    motif_matrix : numpy.ndarray
        Array of shape (n_sequences, n_motifs) containing aggregated motif scores.
    """
    n_motifs = len(fimo_df_list)
    motif_matrix = np.zeros((n_sequences, n_motifs))

    def reverse_complement_attribution(attr_region):
        return attr_region[::-1, ::-1]

    # Process each motif's hits
    for motif_idx, fimo_df in enumerate(fimo_df_list):
        if len(fimo_df) == 0:
            continue

        # Group by sequence for efficiency
        for seq_idx in range(n_sequences):
            # Filter hits for this sequence
            seq_hits = fimo_df[fimo_df["sequence_name"] == seq_idx]

            if len(seq_hits) == 0:
                continue

            # Get scores (potentially weighted by attribution)
            scores = seq_hits["score"].values

            if attributions is not None:
                # Weight each hit by its attribution
                attr_weights = []
                for _, row in seq_hits.iterrows():
                    start, end = row["start"], row["end"]
                    strand = row["strand"]

                    # Extract attribution region
                    attr_region = attributions[seq_idx, :, start:end]

                    # If hit is on negative strand, reverse complement the attribution
                    if strand == "-":
                        attr_region = reverse_complement_attribution(attr_region)

                    # Calculate mean absolute attribution
                    # attr_weight = np.abs(attr_region).mean()
                    attr_weight = (
                        np.abs(attr_region).sum(axis=0).mean()
                    )  # sum over nucleotides, then mean over positions
                    attr_weights.append(attr_weight)

                attr_weights = np.array(attr_weights)
                scores = scores * attr_weights

            # Aggregate
            if aggregate == "max":
                motif_matrix[seq_idx, motif_idx] = scores.max()
            elif aggregate == "sum":
                motif_matrix[seq_idx, motif_idx] = scores.sum()
            elif aggregate == "count":
                motif_matrix[seq_idx, motif_idx] = len(scores)
            elif aggregate == "binary":
                motif_matrix[seq_idx, motif_idx] = 1.0

    return motif_matrix
