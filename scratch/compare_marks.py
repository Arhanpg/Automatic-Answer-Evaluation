import json

REAL_DATA = {
    's1': {'name': 'STUDENT_s1', 'reg': 'ANON_REG_s1', 'sl': 'ANON_SL_s1', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's2': {'name': 'STUDENT_s2', 'reg': 'ANON_REG_s2', 'sl': 'ANON_SL_s2', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's3': {'name': 'STUDENT_s3', 'reg': 'ANON_REG_s3', 'sl': 'ANON_SL_s3', 'q1': 7.0, 'q2': 5.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's4': {'name': 'STUDENT_s4', 'reg': 'ANON_REG_s4', 'sl': 'ANON_SL_s4', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 14.0},
    's5': {'name': 'STUDENT_s5', 'reg': 'ANON_REG_s5', 'sl': 'ANON_SL_s5', 'q1': 4.0, 'q2': 0.0, 'q3': 0.0, 'q4': 1.0, 'tot': 5.0},
    's6': {'name': 'STUDENT_s6', 'reg': 'ANON_REG_s6', 'sl': 'ANON_SL_s6', 'q1': 4.0, 'q2': 2.0, 'q3': 3.0, 'q4': 3.0, 'tot': 12.0},
    's7': {'name': 'STUDENT_s7', 'reg': 'ANON_REG_s7', 'sl': 'ANON_SL_s7', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 1.0, 'tot': 12.0},
    's8': {'name': 'STUDENT_s8', 'reg': 'ANON_REG_s8', 'sl': 'ANON_SL_s8', 'q1': 6.0, 'q2': 6.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's9': {'name': 'STUDENT_s9', 'reg': 'ANON_REG_s9', 'sl': 'ANON_SL_s9', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 1.0, 'tot': 18.0},
    's10': {'name': 'STUDENT_s10', 'reg': 'ANON_REG_s10', 'sl': 'ANON_SL_s10', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's11': {'name': 'STUDENT_s11', 'reg': 'ANON_REG_s11', 'sl': 'ANON_SL_s11', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 2.0, 'tot': 13.0},
    's12': {'name': 'STUDENT_s12', 'reg': 'ANON_REG_s12', 'sl': 'ANON_SL_s12', 'q1': 3.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 5.0},
    's13': {'name': 'STUDENT_s13', 'reg': 'ANON_REG_s13', 'sl': 'ANON_SL_s13', 'q1': 5.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 7.0},
    's14': {'name': 'STUDENT_s14', 'reg': 'ANON_REG_s14', 'sl': 'ANON_SL_s14', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's15': {'name': 'STUDENT_s15', 'reg': 'ANON_REG_s15', 'sl': 'ANON_SL_s15', 'q1': 8.0, 'q2': 0.0, 'q3': 1.0, 'q4': 1.0, 'tot': 10.0},
    's16': {'name': 'STUDENT_s16', 'reg': 'ANON_REG_s16', 'sl': 'ANON_SL_s16', 'q1': 7.0, 'q2': 6.0, 'q3': 2.0, 'q4': 0.0, 'tot': 15.0},
    's17': {'name': 'STUDENT_s17', 'reg': 'ANON_REG_s17', 'sl': 'ANON_SL_s17', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's18': {'name': 'STUDENT_s18', 'reg': 'ANON_REG_s18', 'sl': 'ANON_SL_s18', 'q1': 2.0, 'q2': 2.0, 'q3': 2.0, 'q4': 1.0, 'tot': 7.0},
    's19': {'name': 'STUDENT_s19', 'reg': 'ANON_REG_s19', 'sl': 'ANON_SL_s19', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 2.0, 'tot': 19.0},
    's20': {'name': 'STUDENT_s20', 'reg': 'ANON_REG_s20', 'sl': 'ANON_SL_s20', 'q1': 3.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 4.0},
    's21': {'name': 'STUDENT_s21', 'reg': 'ANON_REG_s21', 'sl': 'ANON_SL_s21', 'q1': 6.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 7.0},
    's22': {'name': 'STUDENT_s22', 'reg': 'ANON_REG_s22', 'sl': 'ANON_SL_s22', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 14.0},
    's23': {'name': 'STUDENT_s23', 'reg': 'ANON_REG_s23', 'sl': 'ANON_SL_s23', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 0.0, 'tot': 12.0},
    's24': {'name': 'STUDENT_s24', 'reg': 'ANON_REG_s24', 'sl': 'ANON_SL_s24', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 14.0},
    's25': {'name': 'STUDENT_s25', 'reg': 'ANON_REG_s25', 'sl': 'ANON_SL_s25', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 15.0},
    's26': {'name': 'STUDENT_s26', 'reg': 'ANON_REG_s26', 'sl': 'ANON_SL_s26', 'q1': 8.0, 'q2': 4.0, 'q3': 3.0, 'q4': 1.0, 'tot': 16.0},
    's27': {'name': 'STUDENT_s27', 'reg': 'ANON_REG_s27', 'sl': 'ANON_SL_s27', 'q1': 7.0, 'q2': 2.0, 'q3': 3.0, 'q4': 1.0, 'tot': 13.0},
    's28': {'name': 'STUDENT_s28', 'reg': 'ANON_REG_s28', 'sl': 'ANON_SL_s28', 'q1': 0.0, 'q2': 2.0, 'q3': 0.0, 'q4': 0.0, 'tot': 2.0},
    's29': {'name': 'STUDENT_s29', 'reg': 'ANON_REG_s29', 'sl': 'ANON_SL_s29', 'q1': 8.0, 'q2': 2.0, 'q3': 2.0, 'q4': 3.0, 'tot': 15.0},
    's30': {'name': 'STUDENT_s30', 'reg': 'ANON_REG_s30', 'sl': 'ANON_SL_s30', 'q1': 8.0, 'q2': 0.0, 'q3': 1.0, 'q4': 0.0, 'tot': 9.0},
    's31': {'name': 'STUDENT_s31', 'reg': 'ANON_REG_s31', 'sl': 'ANON_SL_s31', 'q1': 8.0, 'q2': 0.0, 'q3': 2.0, 'q4': 0.0, 'tot': 10.0},
    's32': {'name': 'STUDENT_s32', 'reg': 'ANON_REG_s32', 'sl': 'ANON_SL_s32', 'q1': 8.0, 'q2': 6.0, 'q3': 3.0, 'q4': 3.0, 'tot': 20.0},
    's33': {'name': 'STUDENT_s33', 'reg': 'ANON_REG_s33', 'sl': 'ANON_SL_s33', 'q1': 6.0, 'q2': 4.0, 'q3': 2.0, 'q4': 0.0, 'tot': 12.0},
    's34': {'name': 'STUDENT_s34', 'reg': 'ANON_REG_s34', 'sl': 'ANON_SL_s34', 'q1': 8.0, 'q2': 0.0, 'q3': 3.0, 'q4': 1.0, 'tot': 12.0},
    's35': {'name': 'STUDENT_s35', 'reg': 'ANON_REG_s35', 'sl': 'ANON_SL_s35', 'q1': 8.0, 'q2': 0.0, 'q3': 2.0, 'q4': 1.0, 'tot': 11.0},
    's36': {'name': 'STUDENT_s36', 'reg': 'ANON_REG_s36', 'sl': 'ANON_SL_s36', 'q1': 6.0, 'q2': 2.0, 'q3': 0.0, 'q4': 2.0, 'tot': 10.0},
    's37': {'name': 'STUDENT_s37', 'reg': 'ANON_REG_s37', 'sl': 'ANON_SL_s37', 'q1': 8.0, 'q2': 2.0, 'q3': 3.0, 'q4': 2.0, 'tot': 15.0}
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
