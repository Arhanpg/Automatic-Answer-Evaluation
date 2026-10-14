from backend import ocr_engine as oe
try:
    imgs = oe._pdf_to_images(r"F:\Automatic answer evaluation dataset\All PDF's\s1.pdf")
    print(imgs)
except Exception as e:
    import traceback
    traceback.print_exc()
