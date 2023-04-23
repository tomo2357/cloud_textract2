#%%
from Library import *
import warnings

warnings.simplefilter("ignore")

# Get the number of tables and the image path
n_table, read_image_path = sys.argv[-2:]
n_table = int(n_table)


# Enter Azure subscription_key and endpoint
azure_ocr_client = AzureOCR(
    "your_subscription_key",
    "your_endpoint",
)
# Enter AWS access_key, secret_access_key, and region
aws_ocr_client = AwsOCR(
    "your_aws_access_key_id", 
    "your_ws_secret_access_key", 
    "your_region"
)

# %%
# Read the image
read_image_path = str(Path(read_image_path).absolute())
print(read_image_path)
img = cv2.imread(read_image_path)

# %%
# Add image processing here if needed and reassign to img
"""
# ここに画像処算を追加する

"""


# %%
# 埋め込み用のndarray

# Create dictionaries and arrays to store information
number_dict = defaultdict(dict)
number_text_arrays = {}
results = {}

# Process the image using both Azure and AWS OCR clients
for name in ("azure", "aws"):
    # cloud毎のclient
    client = eval(f"{name}_ocr_client")
    # numberを入れておくarray
    number_array = np.zeros(list(img.shape[:2]), int)
    # cloudにimageを送り、解析結果を得る
    result = client.read_local_file(read_image_path)
    results[name] = result

    # ndarrayに読み取った値のハッシュを埋め込んでいく
    # 座標とtextを一つずつ取り出す。
    # Store the detected text and their bounding boxes
    for num, (bounding_box, text) in enumerate(
        client.get_bounding_box_and_text(result).items()
    ):
        # azureで伸ばし文字だけだったら弾く
        if name == "azure" and not (set(text) - set("―ー‐—‐ー⁻-")):
            continue
        # number_dict辞書に値を格納
        number_dict[name][num + 1] = (bounding_box, text)
        # 各hashの座標を入れるndarrayでboundingbox内を塗りつぶす
        tmp_array = np.zeros_like(number_array, np.uint8)
        cv2.drawContours(
            tmp_array, [np.array(bounding_box).reshape(4, 2)], 0, color=1, thickness=-1
        )
        # cloud毎のnumber_arrayにnumber値を格納
        number_array = np.where(number_array == 0, tmp_array * (num + 1), number_array)
    number_text_arrays[name] = number_array


#%%

# Get the number arrays for Azure and AWS
azure_number_array = number_text_arrays["azure"]
aws_number_array = number_text_arrays["aws"]

# Find overlapping areas in the number arrays
overlap_azure_array = np.where(
    (azure_number_array != 0) & (aws_number_array != 0), azure_number_array, 0
)
overlap_aws_array = np.where(
    (azure_number_array != 0) & (aws_number_array != 0), aws_number_array, 0
)
# 重なっているbounding_boxについて、それぞれのnumberの組み合わせ
# の集合を得る。
overlap_num = set(
    [
        (azure_num, aws_num)
        for azure_num, aws_num in zip(
            overlap_azure_array.ravel(), overlap_aws_array.ravel()
        )
        if azure_num != 0 and aws_num != 0
    ]
)


# %%

# 重なった集合のうち、azureで日本語又はawsで数字以外が入っていた
# ものはazure採用
# それ以外はaws採用
number_dict_without_overlap = deepcopy(number_dict)
# drop_azure_number = set()
# drop_aws_number = set()

for azure_num, aws_num in overlap_num:
    # if azure_num in drop_azure_number or aws_num in drop_aws_number:
    #    continue
    text_azure = number_dict["azure"][azure_num][1]
    text_aws = number_dict["aws"][aws_num][1]
    if (
        is_japanese(text_azure)
        or "BOD" in re.sub("\s", "", text_azure).upper()
        or "SS" in re.sub("\s", "", text_azure).upper()
        or "MG/L" in re.sub("\s", "", text_azure).upper()
        or "X" in text_azure
        or "×" in text_azure
    ):

        # drop_aws_number.add(aws_num)
        try:
            del number_dict_without_overlap["aws"][aws_num]
        except:
            pass
    else:
        # drop_azure_number.add(azure_num)
        try:
            del number_dict_without_overlap["azure"][azure_num]
        except:
            pass
boundingbox2textWithoutOverlap = dict(
    list(number_dict_without_overlap["azure"].values())
    + list(number_dict_without_overlap["aws"].values())
)

#%% azureとawsを書き込む

img2 = azure_ocr_client.drawBoundingBox(
    azure_ocr_client.get_bounding_box_and_text(results["azure"]), img, (0, 0, 255)
)
img3 = aws_ocr_client.drawBoundingBox(
    aws_ocr_client.get_bounding_box_and_text(results["aws"]), img2, (0, 255, 0)
)
cv2.imwrite(read_image_path.replace(".jpg", "_withBoundingBox.jpg"), img3)


# imshow(img3)


# %%
# ここに文字列の処理を書く
# 平均、最大、最小やT-BODやC-BODがくっついたものは分離させる

boundingbox2textSeparate = deepcopy(boundingbox2textWithoutOverlap)
for bounding_box, text in boundingbox2textWithoutOverlap.items():
    if (
        len(re.findall("BOD", text)) > 1
        or len(re.findall("平均", text))
        + len(re.findall("最大", text))
        + len(re.findall("最小", text))
        > 1
    ):
        print(text)
        text = re.sub("\|", " ", text)
        for pre, post in\
            itertools.permutations(['最大', '平均', '最小'],2):
            text = re.sub(pre+post, pre+' '+post, text)
        separate_text = text.split(" ")
        num_text = len(separate_text)
        # bounding_box = np.array(bounding_box).reshape(4,2)
        top_separate = (
            np.linspace(bounding_box[0:2], bounding_box[2:4], 
                        num=num_text + 1)
            .round()
            .astype(int)
        )
        bottom_separate = (
            np.linspace(bounding_box[6:8], bounding_box[4:6],
                        num=num_text + 1)
            .round()
            .astype(int)
        )
        for i, ar in enumerate(
            zip(top_separate, top_separate[1:], bottom_separate[1:], bottom_separate)
        ):
            new_bounding_box = tuple(np.hstack(ar))
            boundingbox2textSeparate[new_bounding_box] = separate_text[i]

        del boundingbox2textSeparate[bounding_box]


#%%
# imshow(azure_ocr_client.drawBoundingBox(boundingbox2textSeparate, img))
# %%

# 文字幅と行長を計算する
W, H = [], []
for bounding_box, text in boundingbox2textWithoutOverlap.items():
    if len(text):
        w = (bounding_box[2] - bounding_box[0]) / len(text)
        h = bounding_box[-1] - bounding_box[1]
        W.append(w), H.append(h)

w_median, h_median = (np.median(W), np.median(H))

# %%
# boundingbox、text、X,Y,categoryのdataframeを作る
boundingbox2textDf = pd.DataFrame(
    columns=[
        "bounding_box",
        "text",
        "X",
        "Y",
        "category_table",
        "category_columns",
        "category_index",
    ]
)
# boundingboxの中心座標などをdataframeに代入
for i, (bounding_box, text) in enumerate(boundingbox2textSeparate.items()):
    bounding_box_mean = np.array(bounding_box).reshape(4, 2).mean(0)
    boundingbox2textDf.loc[i, ["bounding_box", "text", "X", "Y"]] = (
        bounding_box,
        text,
        *bounding_box_mean,
    )


#%%

# 上下の表で分ける(n_table=2の場合)
if n_table > 1:
    model = AgglomerativeClustering(
        n_clusters=n_table, distance_threshold=None, linkage="ward", affinity="euclidean",
    )


    model.fit(boundingbox2textDf[["Y"]])
    boundingbox2textDf["category_table"] = model.labels_
else:
    boundingbox2textDf["category_table"] = 0
# 高さ順にcategory_tableの順番を付け直す
boundingbox2textDf["mean"] = np.nan
for category_table, df in boundingbox2textDf.groupby("category_table"):
    boundingbox2textDf.loc[df.index, "mean"] = df["Y"].mean()
boundingbox2textDf["category_table"] = boundingbox2textDf["mean"].rank(method="dense")


#%% indexで分ける
# table毎にまずは分ける
for category_table, df in boundingbox2textDf.groupby("category_table"):
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=h_median * 2.5,
        linkage="ward",
        affinity="euclidean",
    )

    # Y座標でカテゴリー化
    model.fit(df[["Y"]])
    boundingbox2textDf.loc[df.index, "category_index"] = model.labels_
    df["category_index"] = model.labels_
    # 高さ順にcategory_tableの順番を付け直す
    for category_index, cdf in df.groupby("category_index"):
        df.loc[cdf.index, "mean"] = cdf["Y"].mean()
    boundingbox2textDf.loc[df.index, "category_index"] = df["mean"].rank(method="dense")

#%%
# columnで分ける
for category_table, df in boundingbox2textDf.groupby("category_table"):
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=w_median * 8,
        linkage="ward",
        affinity="euclidean",
    )
    # X座標でカテゴリー化
    model.fit(df[["X"]])
    boundingbox2textDf.loc[df.index, "category_columns"] = model.labels_
    df["category_columns"] = model.labels_

    # 高さ順にcategory_tableの順番を付け直す
    for category_index, cdf in df.groupby("category_columns"):
        df.loc[cdf.index, "mean"] = cdf["X"].mean()
    boundingbox2textDf.loc[df.index, "category_columns"] = df["mean"].rank(
        method="dense"
    )



# %%
w = int(boundingbox2textDf["category_columns"].max())
h = 0
for i, df in boundingbox2textDf.groupby("category_table"):
    h += int(df["category_index"].max())
    if i != 1:
        h += 3


#%%
yy = 0
excel = pd.DataFrame(np.full([h, w], ""), dtype=object)
for t, table_df in boundingbox2textDf.groupby("category_table"):
    if t != 1:
        yy = (
            boundingbox2textDf.query("category_table==@t-1")["category_index"].max() + 3
        )
    for y, index_table in table_df.groupby("category_index"):
        for x, column_table in index_table.groupby("category_columns"):
            x, y, yy = int(x), int(y), int(yy)
            # print(t, y, x, len(column_table["text"]))
            if len(column_table["text"]) > 1:
                text = "\n".join(column_table["text"])

            else:
                text = column_table["text"].item()
            try:
                text = float(re.sub(",", "", re.sub("\s", "", text)))
            except:
                pass
            excel.iloc[yy + y - 1, x - 1] = text


# %%
excel_path = read_image_path.replace(".jpg", ".xlsx")
excel.to_excel(excel_path, index=None, header=None)

azure_text_path = read_image_path.replace(".jpg", "_azure.txt")
with open(azure_text_path, "wt") as f:
    for bounding_box, text in azure_ocr_client.get_bounding_box_and_text(
        results["azure"]
    ).items():
        f.write("{}, {}\n".format(bounding_box, text))
azure_json_path = read_image_path.replace(".jpg", "_azure.json")
with open(azure_json_path, "wt") as f:
    json.dump(results["azure"].as_dict(), f)
aws_text_path = read_image_path.replace(".jpg", "_aws.txt")
with open(aws_text_path, "wt") as f:
    for bounding_box, text in aws_ocr_client.get_bounding_box_and_text(
        results["aws"]
    ).items():
        f.write("{}, {}\n".format(bounding_box, text))
aws_json_path = read_image_path.replace(".jpg", "_aws.json")
with open(aws_json_path, "wt") as f:
    json.dump(results["aws"], f)

#%%

