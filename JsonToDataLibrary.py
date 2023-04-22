#%%
## logger
import logging, datetime, os
from logging import getLogger, StreamHandler, Formatter, FileHandler, handlers

fmt = Formatter("%(asctime)s %(name)s %(levelname)s %(lineno)s %(message)s")
logger = getLogger("logger")
logger.setLevel(logging.DEBUG)
# logging.disable(logging.NOTSET)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(fmt)
date = str(datetime.date.today()).split(" ")[0]
# file_handler = logging.FileHandler(f"{date}_log.txt")
if not os.path.isdir("./log"):
    os.makedirs("./log")

file_handler = handlers.RotatingFileHandler(
    f"./log/{date}_{logger.name}_rotation_log.txt", maxBytes=10 ** 6, backupCount=1000
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)

# ロガーに追加
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
# logger end


import json, cv2, numba, os, os.path, requests, time
from matplotlib.patches import Polygon
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import boto3, unicodedata, pdf2image, PyPDF2, jaconv, re, sys
from fastprogress import progress_bar
from collections import defaultdict
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import subprocess, traceback, itertools

# %%

#%%
def pdf2JPG(pdf_path, print_page=-1, dpi=200, output_dir=None):
    pdf_path = str(pdf_path)
    reverse_index = pdf_path[::-1].index(".")
    if not output_dir:
        output_dir = pdf_path[: -1 - reverse_index]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    page_count = PyPDF2.PdfFileReader(pdf_path).getNumPages()
    if print_page == -1:
        pages = list(range(1, page_count + 1))
    elif isinstance(print_page, int):
        pages = [print_page]
    else:
        pages = print_page
    image_paths = []
    for page in progress_bar(pages):
        image = pdf2image.convert_from_path(
            pdf_path, dpi, first_page=page, last_page=page
        )[0]

        base_save_name = str(page) + ".jpg"
        save_name = os.path.join(output_dir, base_save_name)
        image.save(save_name, "JPEG")
        logger.info(f"save image {save_name}")
        image_paths.append(save_name)
    return image_paths


def is_digit_only(string):
    flag = False
    for s in re.sub("\s", "", string):
        if s not in "0123456789.,<>-+:":
            return False
        if s in "0123456789":
            flag = True
    if flag:
        return True
    else:
        return False


def is_japanese(string):
    for ch in string:
        name = unicodedata.name(ch)
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


class AzureOCR:
    def __init__(self, subscription_key, endpoint):
        self.read_result = None
        self.client = ComputerVisionClient(
            endpoint, CognitiveServicesCredentials(subscription_key)
        )

    def read_local_file(self, read_image_path):
        with open(read_image_path, "rb") as read_image:
            # Call API with URL and raw response(allows you to get the operation location)
            # read_response = computervision_client.read(read_image_url, raw=True)
            read_response = self.client.read_in_stream(read_image, raw=True)
        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]
        # Call the "GET" API and wait for it to retrieve the results
        while True:
            read_result = self.client.get_read_result(operation_id)
            if read_result.status.lower() not in ["notStarted", "running"]:
                break
            print("Waiting for result")
            time.sleep(5)
        return read_result

    def get_bounding_box_and_text(self, read_result):
        ret_dict = {}
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    bounding_box = [int(p) for p in line.bounding_box]
                    ret_dict[tuple(bounding_box)] = line.text
        return ret_dict

    def drawBoundingBox(self, bounding_box2text, img, color=(0, 255, 0), thickness=1):
        img = img.copy()
        # bounding_box2text = self.get_bounding_box_and_text(read_result)
        for bounding_box in bounding_box2text:
            bounding_box_array = np.array(bounding_box).reshape(4, 2).astype(int)
            cv2.drawContours(img, [bounding_box_array], 0, color, thickness)
        return img


class AwsOCR:
    def __init__(self, aws_access_key_id, aws_secret_access_key, region="us-east-1"):
        self.client = boto3.client(
            "textract",
            region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

    def read_local_file(self, read_image_path):
        img = cv2.imread(read_image_path)
        h, w = img.shape[:2]
        with open(read_image_path, "rb") as read_image:
            read_result = self.client.detect_document_text(
                Document={"Bytes": read_image.read()}
            )
        read_result.update({"ImageSize": (w, h)})
        return read_result

    def get_bounding_box_and_text(self, read_result):
        ret_dict = {}
        w, h = read_result["ImageSize"]
        for block in read_result["Blocks"]:
            if block["BlockType"] != "LINE":
                continue
            polygon = block["Geometry"]["Polygon"]
            bounding_box = ()
            for d in polygon:
                for key, p in d.items():
                    if key.upper() == "X":
                        bounding_box += (int(round(p * w, 0)),)
                    else:
                        bounding_box += (int(round(p * h, 0)),)
            ret_dict[bounding_box] = block["Text"]
        return ret_dict

    def drawBoundingBox(self, bounding_box2text, img, color=(0, 255, 0), thickness=1):
        img = img.copy()
        # bounding_box2text = self.get_bounding_box_and_text(read_result)
        for bounding_box in bounding_box2text:
            bounding_box_array = np.array(bounding_box).reshape(4, 2).astype(int)
            cv2.drawContours(img, [bounding_box_array], 0, color, thickness)
        return img


def imshow(img):
    from IPython.display import Image, display

    """ndarray 配列をインラインで Notebook 上に表示する。
    """
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))


def erase_outest_black(img):
    """
    画像の一番外側にある黒い部分を白く塗りつぶす
    img cv2画像　黒字白背景
    """
    contours, hierarchy = cv2.findContours(img, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
    mask = np.zeros_like(img)
    for cnt, hrc in zip(contours, hierarchy[0]):
        if hrc[-1] == -1:
            cv2.drawContours(mask, [cnt], 0, 255, -1)
    ret_img = cv2.bitwise_and(img, mask) + cv2.bitwise_not(mask)
    return ret_img


def getShadowLength(
    img,
    w=None,
    h=None,
    threshold=200,
    threshold_type=cv2.THRESH_BINARY,
    angle_max=10,
    angle_min=-10,
    num_angle_div=101,
    func=None,
):
    """
    画像の傾きを直す関数
    画像について、長さ１のベクトルへの射影を作り、
    その射影長が短い方向が正しい方向（だろう）
    img : 白背景、黒文字のRGB画像
    w : 画像の横幅（指定しなければ画像の横幅を見る）
    h : 画像の縦幅（指定しなければ画像の縦幅を見る）
    threshold : 画像を二値化する際の閾値(default 200) 
    threshold_type : 画像を二値化する際のタイプ
    (default cv2.THRESH_BINARY)
    angle_max : 傾きを修正する角度のmax
    angle_min : 傾きを修正する角度のmin
    num_angle_div : (angle_max - angle_min)を何分割して試すか
    func : 第一引数に二値化後のGray画像を取るような処理関数
    partialかlambda等で与えること推奨
    """
    # 画像をグレイ化したのち、反転して黒背景白文字にする
    img_rg = cv2.cvtColor(255 - img, cv2.COLOR_RGB2GRAY)
    # 画像を二値化する
    _, img_trg = cv2.threshold(img_rg, threshold, 255, threshold_type)
    # funcがnot Noneであれば、関数で処理する
    if func is not None:
        img_trg = func(img_trg)

    # 画像で点が存在する場所を示すベクトルを作る
    point_vector = np.column_stack(np.where(img_trg > 0)[::-1])
    ## point_pos.shape == (num_point, 2) 左がX, 右がY

    # 画像の高さと幅を得る
    if h is None or w is None:
        h, w = img_trg.shape

    # ベクトルの角度は角度の最大と最小の間を分割数で割ったもの
    degrees = np.linspace(angle_min, angle_max, num_angle_div)

    # 影の長さを入れる枠を作っておく
    # 左が角度、右が影の長さ
    shadows = np.zeros([num_angle_div, 2], np.float64)

    # 図形をdegree回転させたときの行列を作る
    # D.shape == (num_angle_div, 2, 2)
    D = np.array(
        [
            [
                [np.cos(degree / 180 * np.pi), -np.sin(degree / 180 * np.pi)],
                [np.sin(degree / 180 * np.pi), np.cos(degree / 180 * np.pi)],
            ]
            for degree in degrees
        ],
        np.float64,
    )

    # 回転行列を位置ベクトルの転置行列と行列積計算する
    rotate_point_vector = np.round(np.dot(D, point_vector.T))

    # degree毎にX軸とY軸それぞれのunique数を数える
    shadows = get_num_unique(rotate_point_vector, shadows)
    # shadow_length 正射影の長さを入れる枠
    # shadow_length.shape == (num_angle_div, 2)
    # 左 degree, 右length(XとYの合算)
    shadow_length = np.zeros_like(shadows)
    shadow_length[:, 0] = degrees
    shadow_length[:, 1] = shadows[:, 0] / w + shadows[:, 1] / h
    shadow_length = shadow_length[np.abs(shadow_length[:, 0]).argsort()]
    shadow_length = shadow_length[shadow_length[:, 1].argsort()]
    return shadow_length


@numba.jit(
    numba.float64[:, :](numba.float64[:, :, :], numba.float64[:, :]), nopython=True
)
def get_num_unique(rotate_pos_vector, ret):
    """
    get_shadow_lengthの補助関数
    射影の長さを得る。
    """
    for d in range(rotate_pos_vector.shape[0]):
        for xy in range(rotate_pos_vector.shape[1]):
            num_unique = len(np.unique(rotate_pos_vector[d, xy]))
            ret[d, xy] = num_unique
    return ret


# %%


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)
