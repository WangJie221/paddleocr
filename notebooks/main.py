from ocr_annotator import OCRAnnotator
import gradio as gr
import os


API_URL = "http://localhost:8012/ocr"
# 创建标注器实例
annotator = OCRAnnotator(API_URL)

def process_image(image):
    """处理上传的图像"""
    # 保存上传的图像
    if isinstance(image, str):
        image_path = image
    else:
        image_path = "uploaded_image.jpg"
        image.save(image_path)
    
    # 执行OCR
    result_image = annotator.perform_ocr(image_path)
    annotation_data = annotator.get_annotation_data()
    
    # 创建数据表格
    data_table = []
    for item in annotation_data:
        data_table.append([item["id"], item["text"], str(item["bbox"])])
    
    return result_image, data_table

def update_text_handler(id, new_text):
    """更新文本处理器"""
    success = annotator.update_text(id, new_text)
    if success:
        result_image = annotator._create_annotation_image()
        annotation_data = annotator.get_annotation_data()
        
        data_table = []
        for item in annotation_data:
            data_table.append([item["id"], item["text"], str(item["bbox"])])
        
        return result_image, data_table, f"文本ID {id} 更新成功"
    else:
        return gr.update(), gr.update(), f"文本ID {id} 更新失败"

def delete_box_handler(id):
    """删除框处理器"""
    success = annotator.delete_box(id)
    if success:
        result_image = annotator._create_annotation_image()
        annotation_data = annotator.get_annotation_data()
        
        data_table = []
        for item in annotation_data:
            data_table.append([item["id"], item["text"], str(item["bbox"])])
        
        return result_image, data_table, f"框ID {id} 删除成功"
    else:
        return gr.update(), gr.update(), f"框ID {id} 删除失败"

def save_annotations_handler():
    """保存标注结果"""
    if not annotator.current_boxes:
        return "没有可保存的标注数据"
    
    output_dir = "annotations"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(annotator.image_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_annotations.json")
    
    json_path, txt_path = annotator.save_annotations(json_path)
    return f"标注结果已保存:\nJSON: {json_path}\nTXT: {txt_path}"

# 创建Gradio界面 :cite[2]:cite[6]
with gr.Blocks(title="OCR数据标注系统") as demo:
    gr.Markdown("# OCR数据标注系统")
    gr.Markdown("上传图片进行OCR识别，然后对识别结果进行标注和编辑")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="上传图片", type="filepath")
            process_btn = gr.Button("执行OCR识别", variant="primary")
            
            with gr.Row():
                text_id_input = gr.Number(label="文本ID", precision=0)
                new_text_input = gr.Textbox(label="新文本内容")
            
            with gr.Row():
                update_text_btn = gr.Button("更新文本")
                delete_box_btn = gr.Button("删除框")
            
            save_btn = gr.Button("保存标注结果", variant="secondary")
        
        with gr.Column():
            image_output = gr.Image(label="识别结果")
            data_table = gr.Dataframe(
                headers=["ID", "识别文本", "坐标框"],
                label="识别结果表格"
            )
            message_output = gr.Textbox(label="操作消息", interactive=False)
    
    # 事件绑定
    process_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[image_output, data_table]
    )
    
    update_text_btn.click(
        fn=update_text_handler,
        inputs=[text_id_input, new_text_input],
        outputs=[image_output, data_table, message_output]
    )
    
    delete_box_btn.click(
        fn=delete_box_handler,
        inputs=text_id_input,
        outputs=[image_output, data_table, message_output]
    )
    
    save_btn.click(
        fn=save_annotations_handler,
        outputs=message_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)