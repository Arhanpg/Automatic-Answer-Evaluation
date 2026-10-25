import json

REAL_DATA = {
    's1': {'name': 'ADARSH S KAMATAGI', 'reg': '23BCS005', 'sl': '116833', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's2': {'name': 'AMRITANSHU ADITYA', 'reg': '23BCS013', 'sl': '116831', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's3': {'name': 'Babhale Rohan Laxmikant', 'reg': '23BCS026', 'sl': '116858', 'q1': 7.0, 'q2': 5.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's4': {'name': 'Bikram Howladar', 'reg': '23BCS033', 'sl': '116837', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 14.0},
    's5': {'name': 'G. Rakesh', 'reg': '23BCS049', 'sl': '116870', 'q1': 4.0, 'q2': 0.0, 'q3': 0.0, 'q4': 1.0, 'tot': 5.0},
    's6': {'name': 'Ganesh Kumatole', 'reg': '23BCS050', 'sl': '116883', 'q1': 4.0, 'q2': 2.0, 'q3': 3.0, 'q4': 3.0, 'tot': 12.0},
    's7': {'name': 'G. Srivarshith Rao', 'reg': '23BCS051', 'sl': '116844', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 1.0, 'tot': 12.0},
    's8': {'name': 'K. Shriyans', 'reg': '23BCS064', 'sl': '116864', 'q1': 6.0, 'q2': 6.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's9': {'name': 'K. Hemadithya', 'reg': '23BCS068', 'sl': '116875', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 1.0, 'tot': 18.0},
    's10': {'name': 'Kothapally Anjan Reddy', 'reg': '23BCS071', 'sl': '116815', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's11': {'name': 'Krishna Sai Gollamudi', 'reg': '23BCS074', 'sl': '116789', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 2.0, 'tot': 13.0},
    's12': {'name': 'Mohammad Samad Khan', 'reg': '23BCS082', 'sl': '116783', 'q1': 3.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 5.0},
    's13': {'name': 'NUTHAN.K', 'reg': '23BCS088', 'sl': '116827', 'q1': 5.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 7.0},
    's14': {'name': 'P. Shasank Pawan', 'reg': '23BCS096', 'sl': '116825', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's15': {'name': 'Piyush Maurya', 'reg': '23BCS097', 'sl': '116786', 'q1': 8.0, 'q2': 0.0, 'q3': 1.0, 'q4': 1.0, 'tot': 10.0},
    's16': {'name': 'Thrivikram. P', 'reg': '23BCS100', 'sl': '116773', 'q1': 7.0, 'q2': 6.0, 'q3': 2.0, 'q4': 0.0, 'tot': 15.0},
    's17': {'name': 'S. Keerthivassan', 'reg': '23BCS114', 'sl': '116775', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's18': {'name': 'Biddhaastho BS', 'reg': '23BCS127', 'sl': '116838', 'q1': 2.0, 'q2': 2.0, 'q3': 2.0, 'q4': 1.0, 'tot': 7.0},
    's19': {'name': 'SUVAN GURURAJ', 'reg': '23BCS129', 'sl': '116835', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 2.0, 'tot': 19.0},
    's20': {'name': 'A. Guru Sai Harsha', 'reg': '23BDS005', 'sl': '116819', 'q1': 3.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 4.0},
    's21': {'name': 'B. Bharath Kumar', 'reg': '23BDS012', 'sl': '116818', 'q1': 6.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 7.0},
    's22': {'name': 'Chaitra V. Kathimani', 'reg': '23BDS026', 'sl': '116797', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 14.0},
    's23': {'name': 'Gompa Kuladeep', 'reg': '23BDS021', 'sl': '116801', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 0.0, 'tot': 12.0},
    's24': {'name': 'Kala Surya Narayana Rao', 'reg': '23BDS025', 'sl': '116804', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 14.0},
    's25': {'name': 'Kamal Nayan Kumar', 'reg': '23BDS026', 'sl': '116793', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 15.0},
    's26': {'name': 'MANIKESH KUMAR', 'reg': '23BDS032', 'sl': '116807', 'q1': 8.0, 'q2': 4.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's27': {'name': 'SADAN. K', 'reg': '23BDS047', 'sl': '116820', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 13.0},
    's28': {'name': 'S. Harshith', 'reg': '23BDS058', 'sl': '116821', 'q1': 0.0, 'q2': 2.0, 'q3': 0.0, 'q4': 0.0, 'tot': 2.0},
    's29': {'name': 'TARAN JAIN', 'reg': '23BDS062', 'sl': '116814', 'q1': 8.0, 'q2': 2.0, 'q3': 2.0, 'q4': 3.0, 'tot': 15.0},
    's30': {'name': 'Tejaswi A.C', 'reg': '23BDS064', 'sl': '116826', 'q1': 8.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 9.0},
    's31': {'name': 'Vinay N. R', 'reg': '23BDS068', 'sl': '116824', 'q1': 8.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 10.0},
    's32': {'name': 'Yash G. Birje', 'reg': '23BDS070', 'sl': '116790', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's33': {'name': 'Akhilesh R. Ghadge', 'reg': '23BEC004', 'sl': '116780', 'q1': 6.0, 'q2': 4.0, 'q3': 2.0, 'q4': 0.0, 'tot': 12.0},
    's34': {'name': 'Neeraj S. Gawane', 'reg': '23BEC031', 'sl': '116806', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 1.0, 'tot': 12.0},
    's35': {'name': 'Ojas Tulshian', 'reg': '23BEC034', 'sl': '116823', 'q1': 8.0, 'q2': 0.0, 'q3': 2.0, 'q4': 1.0, 'tot': 11.0},
    's36': {'name': 'Ritu Srivastava', 'reg': '25PHDAS006', 'sl': '116750', 'q1': 6.0, 'q2': 2.0, 'q3': 0.0, 'q4': 2.0, 'tot': 10.0},
    's37': {'name': 'PRAVEEN HONGAL', 'reg': '26PHDCS07', 'sl': '116999', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 15.0}
}

with open('data/ground_truth/verified_transcriptions.json', 'r', encoding='utf-8') as f:
    old_data = json.load(f)

print(f"{'Roll':<6} | {'Old Name':<15} -> {'Real Name':<25} | {'Old Tot':<7} -> {'Real Tot':<8} | Discrepancy")
print('-' * 85)
discrepancies = 0
for sid in sorted(REAL_DATA.keys(), key=lambda x: int(x[1:])):
    old_item = old_data.get(sid, {})
    old_name = old_item.get('student_name', 'N/A')
    real_name = REAL_DATA[sid]['name']
    old_tot = old_item.get('examiner_marks', {}).get('total', 'N/A')
    real_tot = REAL_DATA[sid]['tot']
    diff = ''
    if old_tot != real_tot:
        diff = f'DIFF: was {old_tot}, real {real_tot}'
        discrepancies += 1
    print(f"{sid:<6} | {old_name[:15]:<15} -> {real_name[:25]:<25} | {str(old_tot):<7} -> {str(real_tot):<8} | {diff}")

print(f"\nTotal mark discrepancies: {discrepancies}/37")
