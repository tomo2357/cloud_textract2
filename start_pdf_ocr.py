#%%
from JsonToDataLibrary import *
import warnings


warnings.simplefilter("ignore")
read_file_info = pd.read_excel("index.xlsx", 
                               index_col=None)

#%%
for i, (nendo, quantity, quality, precision, remarks, dir,
        n_table_quantity, n_table_quality, n_table_precision)\
    in progress_bar(read_file_info.iterrows(),
                    total = len(read_file_info)):
    #if not (1989 <= nendo <= 1994):
    #    continue
    read_pdf_path = os.path.join(
        dir, f'{nendo}.pdf')
    print_page = defaultdict(list)
    for kind in ('quantity','quality', 'precision'):
        for page in eval(kind).split(','):
            if '-' in page:
                start = int(page.split('-')[0])
                end = int(page.split('-')[1])
                for page in range(start, end+1):
                    print_page[kind].append(page)
            elif not re.sub('\s', '', page):
                continue
            else:
                print_page[kind].append(int(page))
    for kind in ('quantity','quality', 'precision'):
        n_table = str(eval('n_table_'+kind))
        image_paths = pdf2JPG(read_pdf_path, 
                            print_page=print_page[kind],
                            dpi=200)
    
        for img_pth in progress_bar(image_paths):
            abs_path = Path(img_pth).absolute().__str__()
            abs_path_xlsx = abs_path.replace('.jpg', '.xlsx')
            if os.path.exists(abs_path_xlsx):
                logger.info(
                    f"{abs_path_xlsx} is existed. skip!!")
                continue
            image = cv2.imread(img_pth)
            h, w = image.shape[:2]
            tilt_angle = getShadowLength(image)[0, 0]
            M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-tilt_angle, scale=1)
            image_rotated = cv2.warpAffine(image, M, (w, h))
            cv2.imwrite(img_pth, image_rotated)
            for t in range(3):
                try:
                        
                    logger.info(
                        f"subprocess call n_table : {n_table},"
                        f"abs_path : {abs_path}")
                    subprocess.call(
                        [
                            "python3",
                            "-t",
                            "./JsonToData.py",
                            n_table,
                            abs_path,
                        ],
                        shell=False,
                    )
                    logger.info(
                        f"subprocess end")
                    
                    break
                except Exception as e:
                    traceback.print_exc()
                    if t <= 2:
                        print(f"Try : {t+1} / 3")
                        time.sleep(10)

    print("---- END -----")
    read_file_info.loc[i, "end"] = True
    read_file_info.to_excel("nenpo/pdfAndPage.xlsx", index=None)


# %%

# %%
