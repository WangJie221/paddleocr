import numpy as np
import json
import cv2
import base64
import requests
from PIL import Image, ImageDraw, ImageFont


class OCRAnnotator:
    def __init__(self, api_url):
        self.current_boxes = []
        self.current_texts = []
        self.current_image = None
        self.image_path = None
        self.api_url = api_url

    def perform_ocr(self, image_path):
        """执行OCR识别"""
        self.image_path = str(image_path)
        self.current_image = cv2.imread(image_path)

        # 调用PaddleOCR进行识别
        with open(image_path, "rb") as file:
            img = base64.b64encode(file.read()).decode("ascii")
        # payload = {"file": file_data, "fileType": 1}
        payload = {"file": img, "fileType": 1, "visualize": False, "useDocUnwarping": False}
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()

        result = response.json()['result']['ocrResults'][0]['prunedResult']
        self.current_boxes = result['dt_polys']  # 四个点坐标
        # self.current_boxes = result['rec_boxes']  # 文本框坐标(x,y,w,h)
        self.current_texts = result['rec_texts']  # 识别文本
        self.confidences = result['rec_scores']  # 置信度
        
        return self._create_annotation_image()
    
    def _create_annotation_image(self):
        """创建带标注框的图像"""
        if self.current_image is None:
            return None
            
        img_with_boxes = self.current_image.copy()
        
        # 绘制检测框
        for i, box in enumerate(self.current_boxes):
            # 将浮点数坐标转换为整数
            box = np.array(box).astype(np.int32)
            # 绘制多边形框
            cv2.polylines(img_with_boxes, [box], True, (0, 255, 0), 1)
            
            # 添加文本标签
            # text_x = int(np.mean(box[:, 0]))
            # text_y = int(np.mean(box[:, 1]))
            # 指定中文字体
            font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
            font_size = 25
            text_x = int(box[:,0].min())
            text_y = int(box[:, 1].min()) - font_size - 1
            text_y = text_y if text_y >= 0 else int(box[:, 1].max())
            # cv2.putText 中文乱码，改用PIL绘制文本
            # cv2.putText(img_with_boxes, str(self.current_texts[i]), (text_x, text_y), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # 将 OpenCV 图像转换为 PIL 格式
            image_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.truetype(font_path, size=font_size)
            # 绘制文本
            draw.text((text_x, text_y), str(self.current_texts[i]), font=font, fill=(255, 0, 0))

            # 转换回 OpenCV 格式
            img_with_boxes = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # 保存临时结果图像
        output_path = "temp_annotation.jpg"
        cv2.imwrite(output_path, img_with_boxes)
        return output_path
    
    def get_annotation_data(self):
        """获取当前标注数据"""
        data = []
        for i, (box, text) in enumerate(zip(self.current_boxes, self.current_texts)):
            data.append({
                "id": i,
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "text": text
            })
        return data
    
    def update_text(self, id, new_text):
        """更新文本内容"""
        if 0 <= id < len(self.current_texts):
            self.current_texts[id] = new_text
            return True
        return False
    
    def delete_box(self, id):
        """删除标注框"""
        if 0 <= id < len(self.current_boxes):
            self.current_boxes.pop(id)
            self.current_texts.pop(id)
            return True
        return False
    
    def add_box(self, bbox, text):
        """添加新的标注框"""
        self.current_boxes.append(bbox)
        self.current_texts.append(text)
    
    def save_annotations(self, output_path):
        """保存标注结果 :cite[4]:cite[9]"""
        annotation_data = {
            "image_path": self.image_path,
            "annotations": self.get_annotation_data()
        }
        
        # 保存为JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存为TXT格式
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in annotation_data["annotations"]:
                f.write(f"ID: {ann['id']}, Text: {ann['text']}\n")
                f.write(f"BBox: {ann['bbox']}\n")
                f.write("-" * 50 + "\n")
        
        return output_path, txt_path