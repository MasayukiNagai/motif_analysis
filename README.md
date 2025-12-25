# Motif Analysis
- Take attriubtions (N, 4, L) and scans sequences


- MEME: discovers motifs from raw biological sequences
- FIMO: scans sequences given a list of motifs (often from MEME or a database)
- TOMTOM: identifies what those motifs resemble by comparing to a motif database
- TF-Modisco: discovers motifs from attributions


## Pipeline
1. Prepare inputs

1. Get CWM from tfmodisco
2. Scan attribution maps with CWM -> matrix of sequence * motifs


1. Where do the motifs come from?
  - Existing motif database such as JASPAR
  - tmodisco on model attributions
2. What matrix do you scan with?
  - PWM (presence)
  - CWM (usage)
3. What do you scan over?
  -


