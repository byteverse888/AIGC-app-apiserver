# https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API
# python调用sdapi的例子：

import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin

url = "http://127.0.0.1:7860"

payload = {
    "prompt": "1girl, upper body, looking at viewer, overcoat, ..., (photorealistic:1.4), best quality, masterpiece",
    "negative_prompt":"EasyNegative, bad-hands-5, paintings, sketches, (worst quality:2), ..., NSFW, child, childish",
    "steps": 20,
    "sampler_name": "DPM++ SDE Karras",
    "width": 480,
    "height": 640,
    "restore_faces": True
}

response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()

for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    PI = PngImagePlugin.PngInfo()
    PI.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=PI)


# python SDWEBAPI.py

# 下面是算力节点上的脚步参考示例：如何从任务中心获取任务，然后调用本地大模型接口完成AIGC任务，然后提交任务结果和更新任务状态等；
# 算力节点上注册账户是为了激励BTG用的，一个算力节点可以绑定一个账户；
# -*- coding=utf-8

import os
import time
import datetime
import json
import getpass

import requests
from web3 import Web3

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

import GPUtil
from diffusers import DiffusionPipeline
import torch

####################################################################################################
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple web3 requests GPUtil cos-python-sdk-v5 
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers transformers accelerate
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch  #使用nvidia提供的docker镜像预置好torch组件了
# sudo docker run --name  sdxl012001 --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME/sd:/app -it nvcr.io/nvidia/pytorch:23.08-py3  bash
# cd /app/generative-models/btg_ainode && python /app/generative-models/btg_ainode/btg_ainode_clienttask_20231219.py 
# 注意：
# 1、目前仅支持NVIDIA 某些型号，比如T4、A10/A30/A40、RTX30/40系列，建议使用16G以上显存
# 2、wsl + ubuntu22.04 + docker + pytorch2 + cuda12
####################################################################################################

# parse api
API_URL = "http://service-j81m89sf-1302936021.bj.apigw.tencentcs.com"
PARSEAPI_URL = API_URL + "/parse"
WEB3_BTG_URL = API_URL + "/bteth"
chainID = 888  # main:1

#Todo：不能用固定的key，需要用parse用户注册后登录
PARSE_APPLICATION_ID = "BTGAPPId"
PARSE_REST_API_KEY= "BTGAPIKEY"
parse_headers = {
    "X-Parse-Application-Id": "%s" % PARSE_APPLICATION_ID,
    "X-Parse-REST-API-Key": "%s" % PARSE_REST_API_KEY,
    "X-Parse-Revocable-Session": "1",
    "Content-Type": "application/json"
}

# cos config
# Todo: get config from api-server
region = 'ap-beijing'
bucket = 'btgapp1-1302936021'
secret_id = "AKIDZDeSAiU7yjIl16D9VuBHAvZ5VHsLl2AI"
secret_key = "WtAQVQBvgMn7HQcr4GKA0gCZM9qQtCJI"
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key)
cosclient = CosS3Client(config)

####################
# ## logging ####
####################
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger('BTG')  
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d[%(funcName)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler = RotatingFileHandler('logs/btg_ainode.log', maxBytes=1024*1024*10, backupCount=10)
handler.encoding = 'utf-8'
handler.setFormatter(log_format)
logger.addHandler(handler)

#########################################################
###########BTG chain#####################################
#########################################################
w3 = Web3(Web3.HTTPProvider(WEB3_BTG_URL, request_kwargs={'timeout': 15}))
logger.info("web3 is connect:%s, %s", w3.is_connected(), WEB3_BTG_URL)
if w3.is_connected() == False:
    logger.error("web3 is not connect:%s, %s", w3.is_connected(), WEB3_BTG_URL)
    exit(601)

logger.info("load LMM model start")
# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8
logger.info("load LMM model end")

# 调用web3接口创建一个eth账户
def create_btg_account():
    account = w3.eth.account.create()
    btg_account = "new account:%s private_key:%s" % (account.address, account.key.hex())
    # 打印账户和私钥
    print("一定要注意，创建的账户私钥信息一定要保密好哦，账户地址用于BTG激励使用！私钥只会显示一次，如果忘记无法找回！")
    print(btg_account)
    logger.info(btg_account)
    return account.address

# 调用parse API接口登录一个账户并更新用户的account信息
def login_parse(username,password):
    # 登录账户
    payload = json.dumps({"username": username, "password": password})
    request = requests.post(PARSEAPI_URL + "/login", payload, headers=parse_headers)
    if request.status_code == 200:
        parseuser = request.json()
        logger.info("login success: %s" % parseuser)
        print("%s login success" % username)
        # 更新user的ACL以方便后续update用户的属性(Todo: write不需要)
        value =  json.dumps({"ACL": {"*": {"read": True}}})
        ret = update_parse_user(parseuser, value)
        return parseuser
    else:
        logger.error("username %s login failed", username)
        print("%s login failed" % username)
        return ""
    
# 调用parse API接口注册一个账户
def register_parse_user(username,password):
    # 注册账户
    payload = json.dumps({"username": username, "password": password})
    request = requests.post(PARSEAPI_URL + "/users", payload, headers=parse_headers)
    if request.status_code == 200 or request.status_code == 201:
        # Todo: 激活邮件进行验证
        logger.info("register success: %s" % request.text)
        parseuser = request.json()
        # 注册完马上登录
        parseuser = login_parse(username,password)
        return parseuser
    else:
        logger.error("register failed:[%s]" % request.text)
        print("register failed:[%s]" % request.text)
        return ""

# 调用parse API更新用户的account信息
def update_parse_user(parseuser, value):
    userid = parseuser["objectId"]
    payload = value
    new_header = parse_headers
    if parseuser.get("sessionToken"):
        new_header["X-Parse-Session-Token"] = parseuser["sessionToken"]
        
    request = requests.put(PARSEAPI_URL + "/users/%s" % userid, payload, headers=new_header)
    if request.status_code == 200:
        logger.debug("update user account success: %s %s" % (userid, value))
        return True
    else:
        ret_json = request.json()
        logger.error("update user failed: %s %s, reason:%s" % (userid, value, ret_json))
        return False

# 判断账户是否已经存在
def check_parse_user_exist(username):
   # 判断用户是否已经存在，不存在则创建并邮箱激活
    params = {
        "where": '{"username": "%s"}' % username
    }
    request = requests.get(PARSEAPI_URL + "/users", params=params, headers=parse_headers)
    if request.status_code == 200:
        logger.debug("check user exist success: %s" % request.text)
        response = request.json()
        if len(response['results']) > 0:
            parseuser = response['results'][0]
            return parseuser
        else:
            logger.error("check user exist failed:[%s]" % request.text)
            return ""
    else:
        logger.error("check user exist failed:[%s]" % request.text)
        return ""

def register_login_parse_user():
    # 获取用户输入的用户名/邮箱和密码
    username = input("请输入用户名/手机号/邮箱：")
    if len(username) < 3:
        logger.error("用户名长度小于3，退出登录")
        print("用户名长度小于3，退出登录")
        return ""
    # 判断用户是否已经存在
    parseuser = check_parse_user_exist(username)
    if not parseuser:
        # 不存在则提示是否注册新用户
        print("用户不存在，是否注册新用户？(Y/N)")
        while True:
            choice = input().lower()
            if choice == "y":
                break
            elif choice == "n":
                logger.error("用户不存在，退出登录")
                print("用户不存在，退出登录")
                return ""
            else:
                print("输入错误，请重新输入")

        password = getpass.getpass("请输入您的密码：")
        if password != getpass.getpass("请再次输入您的密码："):
            logger.error("密码不一致，退出登录")
            print("密码不一致，退出登录")
            return ""
        parseuser = register_parse_user(username,password)
        if not parseuser:
            logger.error("login and register all failed and exit now")
            print("login and register all failed and exit now")
            return ""
    else:  
        password = getpass.getpass("用户已存在，请输入您的密码：")
        # 存在则登录，如果登录失败则退出
        parseuser = login_parse(username,password)
        if not parseuser:
            logger.error("login failed and exit now")
            print("login failed and exit now")
            return ""

    # 登录成功判断用户web3账户是否存在，不存在则创建web3账户并更新到用户信息中
    if not parseuser.get("account"):
        account = create_btg_account()
        value = json.dumps({
            "account":{
                "__op": "AddUnique",
                "objects": [
                    "%s" % account
                ]
            }
        })
        ret = update_parse_user(parseuser, value)
        if ret:
            logger.debug("update user account success: %s %s" % (username, value))
            print("update user account success: %s %s" % (username, value))
            parseuser['account'] = account
        else:
            logger.error("update user account failed: %s %s" % (username, value))
        return parseuser
    else:
        logger.debug("%s account %s" % (username, parseuser['account']))
        print("%s account %s" % (username, parseuser['account']))
        return parseuser
    
# Query a queue task
def query_queue_task():
    # query没有处理过的posts,获取生产者信息
    params = {
        "limit": 5,
        "skip": 0,
        "where": '{"status": 0}' #status=0表示queue
    }
    request = requests.get(PARSEAPI_URL + "/classes/AIPost", params=params, headers=parse_headers)

    if request.status_code == 200:
        logger.debug("get posts success: %s" % request.text)
        posts = request.json().get("results", [])
        return posts
    else:
        logger.error("get posts failed" % request.text)
        return []
    
# Update task status
def update_task(task_id, data):
    # 更新post状态
    request = requests.put(PARSEAPI_URL + "/classes/AIPost/%s" % task_id, json.dumps(data), headers=parse_headers)
    if request.status_code == 200:
        logger.debug("%s process post success: %s" % (task_id,request.text))
    else:
        logger.error("process post failed: %s" % request.text)


# upload file to cos
def upload_file_tocloud(bucketname, remotename, filename):
    response = cosclient.upload_file(
        Bucket=bucketname,
        LocalFilePath=filename,
        Key=remotename,
        PartSize=1,
        MAXThread=10,
        EnableMD5=False
    )
    img_url = "http://{0}.cos.{1}.myqcloud.com/{2}".format(bucket, region, remotename)
    logger.debug("uploade cos success: %s %s" % (response['ETag'], img_url))
    return img_url

# Process the task
def process_task(task, parseuser, GPU="NVIDIA T4"):
    prompt = task.get('caption', None)
    if not prompt: #前端尽量保证caption不为空
        prompt = "A majestic lion jumping from a big stone at night"

    save_img = "output/refiner_%s.jpg" % task['objectId']
    #save_img = "output/refiner1.jpg"

    # run task with sdxl refiner LMM
    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    image.save(save_img)
    remotename = parseuser['objectId'] + "/" + task_id + '.jpg'
    
    img_url = upload_file_tocloud(bucket, remotename, save_img)

    #update task status and result
    #Todo：GPUID:3090 SN
    data = {
        'status': 2,
        "imagesUrl":{
            "__op": "AddUnique",
            "objects": [
                "%s" % img_url
            ]
        },
        "createdUser": "%s" % parseuser["objectId"],
        "GPUID": "%s" % GPU
    }
    
    print("task %s process success reulst %s" % (task_id, img_url))
    update_task(task['objectId'], data)

    # free memory
    torch.cuda.empty_cache()

# main 函数
if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()

    # 判断是否存在GPU
    try:
        # 获取所有GPU的信息
        gpus = GPUtil.getGPUs()
        # 遍历每个GPU并打印其型号
        # Todo：如果有多个可用的GPU，需要优化支持多个GPU同时执行task
        if len(gpus) == 0:
            print("No GPU found")
            exit(600)
        for gpu in gpus:
            gpu_name = gpu.name
            GPU = gpu_name
        print(f"GPU: {gpu_name}")
        logger.info(f"GPU: {GPU}")
    except:
        print("No GPU found")
        exit(600)
    
    # 如果有用户相关的环境变量，直接使用环境变量登录
    username = os.environ.get("BTG_USERNAME")
    password = os.environ.get("BTG_PASSWORD")
    # 使用环境登录
    if username and password:
        parseuser = login_parse(username,password)
        if (not parseuser) or (parseuser.get("account") == None):
            logger.error("login_parse failed")
            print("login_parse failed")
            exit(600)
        else:
            print("%s login success and account is %s" % (username, parseuser['account']))
    else:
        # 登录账户或注册新账户
        parseuser = register_login_parse_user()
        if (not parseuser) or (parseuser.get("account") == None):
            logger.error("register_login_parse_user failed or get account failed")
            exit(600)

    # 使用当前账户执行任务（贡献算力，角色是平台生产者）
    while True:
        try:
            print("Start to process queue task %s\n" % (datetime.datetime.now()))
            # get one pending task
            tasks = query_queue_task()
            if not tasks:
                print('Dont have queue task...\n')
                logger.info('Dont have queue task...\n')
                time.sleep(60)  # Wait for 60 seconds before querying the next task
                continue
        
            for one_task in tasks:
                task_id = one_task['objectId']
                #lock task
                data = {'status': 1}
                update_task(task_id, data)
        
                #process task
                try:
                    result = process_task(one_task, parseuser, GPU)
                except Exception as e:
                    data = {'status': 0}
                    update_task(task_id, data)
            
            logger.info("End to process queue task...\n")
            print("End to process queue task %s\n" % (datetime.datetime.now()))
        except Exception as e:
            current_time = datetime.datetime.now()
            print("发生了异常: %s %s " % (current_time, str(e)))
            logger.error("发生了异常: %s %s " % (current_time, str(e)))
