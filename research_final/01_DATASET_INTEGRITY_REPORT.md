# Dataset Integrity and Provenance Report

## 1. Corpus Overview

- **Course**: DS317 / DS353 Large Language Models
- **Institution**: Indian Institute of Information Technology Dharwad (IIIT Dharwad)
- **Exam Format**: Mid-Semester Handwritten Technical Examination
- **Total Booklets**: 37
- **Total Rendered Pages**: 195
- **Split Distribution**: Train (25 students, 67.6%), Validation (6 students, 16.2%), Locked Test (6 students, 16.2%)

## 2. Cryptographic Hashes & Partitioning

| Student ID | Split | PDF File | SHA-256 Checksum | Pages | Rendered Pages | Size (Bytes) |
|------------|-------|----------|------------------|-------|----------------|--------------|
| s1 | train | s1.pdf | `16416dc036bb1807...` | 5 | 1 | 14918023 |
| s2 | validation | s2.pdf | `9304dc9ee2e2741b...` | 8 | 3 | 23065130 |
| s3 | train | s3.pdf | `68a88fd9fe26bba8...` | 5 | 5 | 14260185 |
| s4 | train | s4.pdf | `ad9ce045a042e422...` | 9 | 9 | 26220029 |
| s5 | validation | s5.pdf | `4124c4adc5fa0491...` | 4 | 4 | 11544141 |
| s6 | train | s6.pdf | `7eee65e72747a760...` | 5 | 5 | 15120002 |
| s7 | test | s7.pdf | `535f76e24a19e4b8...` | 5 | 5 | 13982467 |
| s8 | train | s8.pdf | `0bba9a2082f084b6...` | 4 | 4 | 11718515 |
| s9 | train | s9.pdf | `fb3ee2545865816c...` | 4 | 4 | 11690316 |
| s10 | train | s10.pdf | `3e65cdffbebf88ce...` | 8 | 8 | 23908371 |
| s11 | train | s11.pdf | `1d9dfe93443a87f6...` | 9 | 9 | 25520236 |
| s12 | validation | s12.pdf | `b80dc10b29434b68...` | 3 | 3 | 8438481 |
| s13 | train | s13.pdf | `1cd519ec6ede32e9...` | 5 | 5 | 13499367 |
| s14 | test | s14.pdf | `a4930a739acc6cd2...` | 6 | 6 | 17037654 |
| s15 | train | s15.pdf | `0e21716d0127aba0...` | 6 | 6 | 16409173 |
| s16 | train | s16.pdf | `1a343d471beac1b9...` | 3 | 3 | 8524997 |
| s17 | train | s17.pdf | `49cde06998f6004a...` | 7 | 7 | 19632852 |
| s18 | validation | s18.pdf | `9897dabf13d644d9...` | 3 | 3 | 8278810 |
| s19 | train | s19.pdf | `d3642b82b8ff5009...` | 6 | 6 | 17456043 |
| s20 | train | s20.pdf | `00b3fea392b668d8...` | 4 | 4 | 10581675 |
| s21 | validation | s21.pdf | `e681d23e98e8783c...` | 6 | 6 | 15522724 |
| s22 | train | s22.pdf | `e85c55ed022174bf...` | 5 | 5 | 13852079 |
| s23 | train | s23.pdf | `f28c9f972bd97ea5...` | 6 | 6 | 16501335 |
| s24 | test | s24.pdf | `203dab2d0c9d05b1...` | 4 | 4 | 11559406 |
| s25 | train | s25.pdf | `ebc17deccdc22b13...` | 4 | 4 | 11304850 |
| s26 | train | s26.pdf | `ac0f120f11403c9f...` | 4 | 4 | 11526875 |
| s27 | train | s27.pdf | `b61a16df002ae3d1...` | 6 | 6 | 15955132 |
| s28 | train | s28.pdf | `086240bf27217605...` | 2 | 2 | 5387952 |
| s29 | test | s29.pdf | `a851c283ce9c1002...` | 7 | 7 | 18918572 |
| s30 | test | s30.pdf | `192c4ce93cf45a6e...` | 5 | 5 | 14247071 |
| s31 | train | s31.pdf | `3b56f6828e31e1e7...` | 4 | 4 | 10915777 |
| s32 | train | s32.pdf | `d33a246f4e056881...` | 5 | 5 | 13370721 |
| s33 | validation | s33.pdf | `2276cfc26d550bda...` | 4 | 4 | 11398218 |
| s34 | train | s34.pdf | `241f5d876ea06a7b...` | 7 | 7 | 18044684 |
| s35 | train | s35.pdf | `27345cfdf6ddde57...` | 5 | 5 | 13589298 |
| s36 | train | s36.pdf | `c4c912bd2b402ab9...` | 4 | 4 | 10595226 |
| s37 | test | s37.pdf | `3517c477db47ee0a...` | 8 | 8 | 21096432 |

## 3. Zero Data-Leakage Verification

- **Student Isolation**: No student booklet is shared across splits. Partitions are formed strictly at the student booklet level.
- **Locked Test Integrity**: The 6 test booklets (`s7`, `s14`, `s24`, `s29`, `s30`, `s37`) were sequestered prior to model fitting.
- **Feature & Normalization Isolation**: All scaler parameters, conformal quantile cutoffs, and preprocessing parameters are derived strictly from the training partition and validated on validation before application to locked test.
