import json
from pathlib import Path
from PIL import Image

data_root = Path('F:/Automatic answer evaluation dataset')
tm_dir = Path('data/teacher_marks')

census = []
total_pages = 0
for sid_num in range(1, 38):
    sid = f's{sid_num}'
    s_dir = data_root / sid
    imgs = sorted(list(s_dir.glob('*.jpg')))
    total_pages += len(imgs)
    
    tm_file = tm_dir / f'{sid}.json'
    marks = json.load(open(tm_file))['marks'] if tm_file.exists() else {}
    
    orientations = []
    for im_p in imgs:
        with Image.open(im_p) as im:
            orientations.append('landscape' if im.width > im.height else 'portrait')
            
    census.append({
        'sid': sid,
        'page_count': len(imgs),
        'orientations': list(set(orientations)),
        'total_marks': sum(marks.values()),
        'marks': marks
    })

print(f"Total students cataloged: {len(census)}")
print(f"Total pages cataloged: {total_pages}")
rotated_students = [c['sid'] for c in census if 'landscape' in c['orientations']]
print(f"Students with landscape scans ({len(rotated_students)}): {rotated_students}")

print("\nDetailed booklet census:")
for c in census:
    print(f"  {c['sid']}: {c['page_count']} pages, orientations={c['orientations']}, total_marks={c['total_marks']}, Q_marks={c['marks']}")
