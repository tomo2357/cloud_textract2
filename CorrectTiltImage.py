#%%
from Library import *
import warnings

warnings.simplefilter("ignore")

read_image_path,save_image_path = sys.argv[-2:]

read_image_path = str(Path(read_image_path).absolute())
image = cv2.imread(read_image_path)
h, w = image.shape[:2]
tilt_angle = getShadowLength(image)[0, 0]
M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=-tilt_angle, scale=1)
image_rotated = cv2.warpAffine(image, M, (w, h))
cv2.imwrite(save_image_path, image_rotated)
