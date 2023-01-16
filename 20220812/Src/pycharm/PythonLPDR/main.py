import opencv_plate_locate.plate_locator as pl
if __name__ == '__main__':
    plate_image = '' # 遍历获取所有的车牌图片，逐一
    candidate_plates = pl.get_candidate_plates_by_sobel(plate_image)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
