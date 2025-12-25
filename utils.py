import subprocess
import h5py
import numpy as np

# ============================================================================
# Step 2: Run TF-MoDISco-lite
# ============================================================================
def run_tfmodisco(
    seq_path,
    attr_path,
    output_h5="tfmodisco_results.h5",
    output_dir="tfmodisco_report",
    n_seqlets=2000,
    window_size=1000
):
    """Run tfmodisco-lite for motif discovery"""

    # Run modisco motifs
    cmd_motifs = [
        'modisco', 'motifs',
        '-s', seq_path,
        '-a', attr_path,
        '-n', str(n_seqlets),
        '-o', output_h5,
        '-w', str(window_size)
    ]
    print("CMD:", " ".join(cmd_motifs))
    subprocess.run(cmd_motifs, check=True)

    # Generate report
    cmd_report = [
        'modisco', 'report',
        '-i', output_h5,
        '-o', output_dir
    ]
    print("CMD:", " ".join(cmd_report))
    subprocess.run(cmd_report, check=True)

    print(f"TF-MoDISco results saved to {output_h5}")
    print(f"Report saved to {output_dir}")


# ============================================================================
# Step 3: Convert TF-MoDISco results to MEME format
# ============================================================================
def modisco_to_meme(modisco_h5_path, output_meme_path, use_cwm=True, trim_threshold=0.3):
    """
    Convert TF-MoDISco results to MEME format with proper numerical sorting
    """
    import re

    def calculate_ic(matrix):
        """Calculate information content per position"""
        ic = np.sum(matrix * np.log2(matrix + 1e-10), axis=1) + 2
        return ic

    def trim_motif(matrix, threshold=0.3):
        """Trim uninformative flanking positions"""
        prob_matrix = np.abs(matrix) + 0.001
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        ic = calculate_ic(prob_matrix)

        informative = ic > threshold
        if not informative.any():
            return matrix

        indices = np.where(informative)[0]
        start, end = indices[0], indices[-1] + 1

        return matrix[start:end]

    def extract_pattern_number(pattern_name):
        """Extract numeric value from pattern name for sorting"""
        match = re.search(r'pattern_(\d+)', pattern_name)
        if match:
            return int(match.group(1))
        return 0

    with h5py.File(modisco_h5_path, 'r') as f:
        meme_lines = ['MEME version 4\n\n']
        meme_lines.append('ALPHABET= ACGT\n\n')
        meme_lines.append('strands: + -\n\n')

        # Collect all motifs first for global sorting
        all_motifs = []

        # Iterate through metaclusters
        for metacluster in ['pos_patterns', 'neg_patterns']:
            if metacluster not in f:
                continue

            metacluster_group = f[metacluster]

            # Get all pattern names
            pattern_names = list(metacluster_group.keys())

            for pattern_name in pattern_names:
                pattern = metacluster_group[pattern_name]

                # Choose CWM or PWM
                if use_cwm and 'contrib_scores' in pattern:
                    matrix = pattern['contrib_scores'][:]
                    matrix_type = 'CWM'
                elif 'sequence' in pattern:
                    matrix = pattern['sequence'][:]
                    matrix_type = 'PFM'
                else:
                    continue

                all_motifs.append({
                    'metacluster': metacluster,
                    'pattern_name': pattern_name,
                    'matrix': matrix,
                    'matrix_type': matrix_type,
                    'pattern_num': extract_pattern_number(pattern_name)
                })

        # Sort by metacluster (pos first, then neg) and then by pattern number
        all_motifs.sort(key=lambda x: (
            0 if x['metacluster'] == 'pos_patterns' else 1,
            x['pattern_num']
        ))

        # Write sorted motifs
        for motif_info in all_motifs:
            matrix = motif_info['matrix']

            # Trim uninformative flanks
            matrix = trim_motif(matrix, threshold=trim_threshold)

            if use_cwm:
                matrix_abs = np.abs(matrix)
                matrix_norm = matrix_abs / (matrix_abs.sum(axis=1, keepdims=True) + 1e-10)
            else:
                matrix_norm = matrix + 0.001
                matrix_norm = matrix_norm / matrix_norm.sum(axis=1, keepdims=True)

            motif_length = matrix_norm.shape[0]

            # Write motif header
            motif_name = f"{motif_info['metacluster']}_{motif_info['pattern_name']}"
            meme_lines.append(f'MOTIF {motif_name}\n')
            meme_lines.append(f'letter-probability matrix: alength= 4 w= {motif_length}\n')

            # Write matrix
            for position in matrix_norm:
                meme_lines.append('  '.join([f'{p:.6f}' for p in position]) + '\n')

            meme_lines.append('\n')

        print(f"Converted {len(all_motifs)} motifs to MEME format")

    # Write to file
    with open(output_meme_path, 'w') as f:
        f.writelines(meme_lines)

    print(f"MEME file saved to {output_meme_path}")


def get_motifs_from_tfmodisco(modisco_h5_path):
    """
    Export raw CWMs (and PWMs) as numpy arrays for custom scanning
    This preserves the original contribution scores
    Returns an OrderedDict with motifs sorted by metacluster and pattern number
    """
    import re
    from collections import OrderedDict

    def extract_pattern_number(pattern_name):
        """Extract numeric value from pattern name for sorting"""
        match = re.search(r'pattern_(\d+)', pattern_name)
        if match:
            return int(match.group(1))
        return 0

    with h5py.File(modisco_h5_path, 'r') as f:
        # Collect all motifs first
        all_motifs = []

        for metacluster in ['pos_patterns', 'neg_patterns']:
            if metacluster not in f:
                continue

            metacluster_group = f[metacluster]

            # Get and sort pattern names numerically
            pattern_names = list(metacluster_group.keys())

            for pattern_name in pattern_names:
                pattern = metacluster_group[pattern_name]
                motif_name = f"{metacluster}_{pattern_name}"

                all_motifs.append({
                    'name': motif_name,
                    'metacluster': metacluster,
                    'pattern_num': extract_pattern_number(pattern_name),
                    'cwm': pattern['contrib_scores'][:] if 'contrib_scores' in pattern else None,
                    'pfm': pattern['sequence'][:] if 'sequence' in pattern else None,
                })

        # Sort by metacluster (pos first, then neg) and then by pattern number
        all_motifs.sort(key=lambda x: (
            0 if x['metacluster'] == 'pos_patterns' else 1,
            x['pattern_num']
        ))

        # Build ordered dictionary
        motif_data = OrderedDict()
        for motif in all_motifs:
            motif_data[motif['name']] = {
                'cwm': motif['cwm'],
                'pfm': motif['pfm'],
            }

    return motif_data


# ============================================================================
# Step 4: Scan sequences with memelite FIMO
# ============================================================================
from memelite.fimo import fimo
def prepare_fasta(sequences, output_fasta):
    """Write list of sequence strings to FASTA format with max 80 chars per line"""
    max_width = 80

    with open(output_fasta, 'w') as f:
        for idx, seq in enumerate(sequences):
            seq_str = str(seq).strip().upper()
            f.write(f'>seq_{idx}\n')
            for i in range(0, len(seq_str), max_width):
                f.write(seq_str[i:i + max_width] + '\n')

    print(f"FASTA file saved to {output_fasta}")


def scan_sequences_memelite(meme_file, sequences, attributions=None,
                           thresh=1e-4, motif_pseudo=0.1):
    """
    Scan sequences using memelite FIMO

    Args:
        meme_file: Path to MEME format motif file
        sequences: (N, 4, L) one-hot encoded sequences
        attributions: Optional (N, 4, L) attribution array for weighting
        thresh: p-value threshold
        motif_pseudo: Pseudocount for motif matrices

    Returns:
        DataFrame with FIMO results
    """
    # Convert one-hot sequences to strings
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_strings = []

    for seq in sequences:
        seq_str = ''.join([base_map[np.argmax(seq[:, i])] for i in range(seq.shape[1])])
        seq_strings.append(seq_str)

    # Run FIMO using memelite
    print(f"Scanning {len(seq_strings)} sequences with FIMO...")
    results_df = fimo(
        seq_strings,
        meme_file,
        thresh=thresh,
        motif_pseudo=motif_pseudo
    )

    print(f"Found {len(results_df)} motif hits")

    return results_df


# ============================================================================
# Step 5: Build sequence × motif matrix from memelite FIMO results
# ============================================================================

def build_motif_matrix_memelite(fimo_df, n_sequences, n_motifs, motif_names,
                                attributions=None, aggregate='max'):
    """
    Build sequence x motif matrix from memelite FIMO results

    Args:
        fimo_df: DataFrame from memelite FIMO
        n_sequences: Total number of sequences
        n_motifs: Number of motifs
        motif_names: List of motif names (in order)
        attributions: Optional (N, 4, L) attribution array for weighting
        aggregate: How to aggregate multiple hits ('max', 'sum', 'count', 'binary')

    Returns:
        motif_matrix: (N_sequences, N_motifs) matrix
    """
    # Create motif name to index mapping
    motif_to_idx = {name: i for i, name in enumerate(motif_names)}

    # Initialize matrix
    motif_matrix = np.zeros((n_sequences, n_motifs))

    # Process each hit
    for _, row in fimo_df.iterrows():
        seq_idx = int(row['sequence_idx'])
        motif_name = row['motif_id']

        if motif_name not in motif_to_idx:
            continue

        motif_idx = motif_to_idx[motif_name]
        score = row['score']
        start, stop = int(row['start']), int(row['stop'])

        # Optional: weight by attribution
        if attributions is not None:
            # Average absolute attribution in the motif region
            attr_region = attributions[seq_idx, :, start:stop]
            attr_weight = np.abs(attr_region).mean()
            score = score * attr_weight

        # Aggregate scores
        if aggregate == 'max':
            motif_matrix[seq_idx, motif_idx] = max(motif_matrix[seq_idx, motif_idx], score)
        elif aggregate == 'sum':
            motif_matrix[seq_idx, motif_idx] += score
        elif aggregate == 'count':
            motif_matrix[seq_idx, motif_idx] += 1
        elif aggregate == 'binary':
            motif_matrix[seq_idx, motif_idx] = 1

    return motif_matrix


def get_motif_names_from_meme(meme_file):
    """Extract motif names from MEME file"""
    motif_names = []
    with open(meme_file, 'r') as f:
        for line in f:
            if line.startswith('MOTIF '):
                motif_name = line.strip().split()[1]
                motif_names.append(motif_name)
    return motif_names
