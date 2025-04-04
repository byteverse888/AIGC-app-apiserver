# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import requests
import time
import json
from minio import Minio
import base64
import os
from urllib.parse import urljoin

# python task_processor.py txt2audio account123
# python task_processor.py txt2img account123

# 配置日志
logger = logging.getLogger('AITASK')
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d[%(funcName)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'aigc_task.log')
handler = RotatingFileHandler(log_path, maxBytes=1024*1024*10, backupCount=10, encoding='utf-8')
handler.setFormatter(log_format)
logger.addHandler(handler)

os.makedirs('logs', exist_ok=True)
os.makedirs('tmp', exist_ok=True)

# 配置参数
API_URL = "http://82.156.86.71:8082"
TASK_URL = urljoin(API_URL, "/parseapi/parse/classes/AITask")
TASKNUM_ONCE = 2
PARSE_APPLICATION_ID = "BTGAPPId"
PARSE_REST_API_KEY = "BTGAPIKEY"
parse_headers = {
    "X-Parse-Application-Id": PARSE_APPLICATION_ID,
    "X-Parse-REST-API-Key": PARSE_REST_API_KEY,
    "X-Parse-Revocable-Session": "1",
    "Content-Type": "application/json"
}
MINIO_URL = "http://82.156.86.71:9000"
SDAPI = "http://127.0.0.1:7860"
# AUDIOAPI = "http://127.0.0.1:7861"
AUDIO_STT_API = "http://127.0.0.1:8083/v1/audio/transcriptions"
AUDIO_TTS_API = "http://127.0.0.1:8084/v1/audio/speech"

# MinIO配置
minio_client = Minio(
    endpoint="82.156.86.71:9000",
    access_key="7yG6o8Fx5FODZayRkaN6",
    secret_key="NDBKpRdNcauBXweruwkOu4pbqItIcIkYmVlmbCBB",
    secure=False
)

def get_queue_tasks(task_type):
    try:
        params = {
            "limit": TASKNUM_ONCE,
            "skip": 0,
            "where": json.dumps({
                "type": task_type,
                "status": 0
            })
        }
        logger.info(f"开始查询待处理{task_type}任务，查询参数: {params}")
        response = requests.get(TASK_URL, params=params, headers=parse_headers)
        response.raise_for_status()
        results = response.json().get('results', [])
        logger.info(f"成功获取{len(results)}个{task_type}任务")
        return results
    except Exception as e:
        logger.exception(f"获取任务失败，任务类型: {task_type}，错误信息: ")
        return []


# Todo: 后续考虑封装一个上传下载的接口，同时支持COS和Minio，根据配置文件决定使用哪个存储服务
def upload_to_minio(file_data, account_id, task_id, file_name):
    try:
        logger.info(f"开始上传文件到MinIO，任务ID: {task_id} 文件名: {file_name}")
        file_path = f"tmp/{file_name}"
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        minio_client.fput_object(
            "aitask",
            f"{account_id}/{file_name}",
            file_path
        )
        logger.info(f"文件上传成功，存储路径: aitask/{account_id}/{file_name}")
        # os.remove(file_path)
        return urljoin(MINIO_URL, f"aitask/{account_id}/{file_name}")
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise

def update_task_status(task, status, result=None, account_id=None):
    logger.info(f"更新任务状态 任务ID: {task['objectId']} 新状态: {status}")
    try:
        update_url = f"{TASK_URL}/{task['objectId']}"
        update_data = {
            "status": status,
            "result": result or [],
            "executor": account_id if account_id else "unknown"
        }
        response = requests.put(update_url, json=update_data, headers=parse_headers)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"任务状态更新失败: {str(e)}")

def process_txt2img(task, account_id):
    try:
        logger.info(f"开始处理txt2img任务，参数: prompt={task['data']['prompt']}")
        request_data = task['data']
        response = requests.post(f"{SDAPI}/sdapi/v1/txt2img", json=request_data)
        response.raise_for_status()
        logger.info(f"txt2img处理成功，生成{len(response.json()['images'])}张图片")

        images = response.json()['images']
        
        objectId = task['objectId']
        file_urls = []
        for i, img_data in enumerate(images):
            # Base64解码图像数据
            image_bytes = base64.b64decode(img_data)
            file_name = f"output_{objectId}_{i}.png"
            url = upload_to_minio(image_bytes, account_id, objectId, file_name)
            file_urls.append(url)

        update_task_status(task, 1, file_urls, account_id)

        return file_urls
    except Exception as e:
        logger.error(f"text2img处理失败: {str(e)}")
        raise

def process_img2img(task, account_id):
    try:
        logger.info(f"开始处理img2img任务，初始化图片尺寸: {len(task['data']['init_image'])}字节")
        request_data = task['data']
        # if 'init_image' in request_data:
        #     request_data['init_image'] = base64.b64encode(request_data['init_image']).decode('utf-8')

        response = requests.post(f"{SDAPI}/sdapi/v1/img2img", json=request_data)
        response.raise_for_status()
        logger.info(f"img2img处理完成，输出图片尺寸: {len(response.json()['images'][0])}字节")
        return response.json()['images']
    except Exception as e:
        logger.error(f"img2img处理失败: {str(e)}")
        raise

def process_txt2speech(task, account_id):
    try:
        logger.info(f"开始文本转语音处理，文本长度: {len(task['data']['text'])}字符")
        
        request_data = request_data = task['data']
        response = requests.post(f"{AUDIO_TTS_API}", json=request_data)
        response.raise_for_status()        
        audios = response.json()['audio']

        objectId = task['objectId']
        file_urls = []
        for i, audio_data in enumerate(audios):
            # Base64解码数据
            file_name = f"output_{objectId}_{i}.wav"
            audio_data.save(file_name)
            logger.info(f"音频文件已保存到：{file_name}")

            url = upload_to_minio(audio_data, account_id, objectId, file_name)
            logger.info(f"音频文件已上传至：{url}")
            file_urls.append(url)

        update_task_status(task, 1, file_urls, account_id)

        return file_urls

        # return response.json()['audio']
    except Exception as e:
        logger.error(f"text2audio处理失败: {str(e)}")
        raise
    
def process_speech2txt(task, account_id):
    try:
        audio_file = task['data']['input']
        # 从MinIO下载音频文件
        local_file_tmp = os.path.join("tmp", "%s_audio.wav" % task['objectId'])
        if os.path.exists(local_file_tmp):
            os.remove(local_file_tmp)
        
        minio_client.fget_object("aitask", audio_file, local_file_tmp)
        
        # 发送音频文件到音频转文本API
        # Todo: 后续不通过调用API转换，直接调用接口处理，避免API服务持续运行占用资源
        # 说明: 优点是可以处理多种类型任务，不像现在一个边缘节点只能处理一种服务对应的任务
        #      缺点是需要反复加载模型，任务处理时间边长
        try:
            with open(local_file_tmp, 'rb') as audio_file:
                files = {'file': (os.path.basename(local_file_tmp), audio_file, 'audio/wav')}
                response = requests.post(f"{AUDIO_STT_API}", files=files)
                response.raise_for_status()
        finally:
            if os.path.exists(local_file_tmp):
                os.remove(local_file_tmp)
        
        if response.status_code != 200:
            logger.error(f"音频转文本失败: {response.text}")
            raise Exception(f"音频转文本失败: {response.text}")

        text_result = response.json()['text']
        
        # 保存识别结果到临时文件
        file_name = f"{task['objectId']}_stt_result.txt"
        file_path = os.path.join("tmp", file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_result)
        
        # 上传到MinIO
        url = upload_to_minio(open(file_path, 'rb').read(), account_id, task['objectId'], file_name)
        logger.info(f"语音识别结果已保存并上传至MinIO: {url}")

        # 清理临时文件
        os.remove(file_path)

        logger.info(f"任务 {task['objectId']} 处理完成")

        return url

    except Exception as e:
        logger.error(f"audio2txt处理失败: {str(e)}")
        raise

def main_loop(task_type, account_id):
    logger.info(f"启动{task_type}任务处理循环，账户ID: {account_id}")
    while True:
        try:
            tasks = get_queue_tasks(task_type)
            logger.info(f"获取到{len(tasks)}个待处理{task_type}任务")
            if not tasks:
                time.sleep(60)
                continue

            for task in tasks:
                try:
                    logger.info(f"开始处理任务 {task['objectId']} task：{task}")
                    
                    # data是客户端根据用户输入post时的参数
                    if 'data' not in task:
                        logger.error(f"任务 {task['objectId']} 缺少data字段，数据格式错误")
                        continue
                    
                    resulet_urls = []
                    if task_type == "txt2img":
                        url = process_txt2img(task, account_id)
                    elif task_type == "img2img":
                        url = process_img2img(task, account_id)
                    elif task_type == "txt2speech":
                        url = process_txt2speech(task, account_id)
                    elif task_type == "speech2txt":
                        url = process_speech2txt(task, account_id)
                        
                    # Todo: 后续增加comfyUI工作流的处理

                    # 更新任务状态为完成
                    resulet_urls.append(url)
                    update_task_status(task, 1, resulet_urls, account_id)
                    logger.info(f"任务类型 {task_type} {task['objectId']} 处理完成")

                except Exception as e:
                    logger.error(f"任务 {task['objectId']} 处理失败: {str(e)}")

        except Exception as e:
            logger.error(f"主循环异常: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python task_processor.py [task_type] [account_id]")
        print("task_type: txt2img, img2img, txt2speech, speech2txt")
        print("account_id: BTG 地址，用于标识任务的执行者账户信息用于激励发放，如account123，没有可以通过浏览器插件或登录平台注册一个")
        print("Example: python task_processor.py speech2txt account123")
        sys.exit(1)
    main_loop(sys.argv[1], sys.argv[2])