import hashlib
import os

import requests
from alibabacloud_bailian20231229 import models as bailian_20231229_models
from alibabacloud_bailian20231229.client import Client as bailian20231229Client
from alibabacloud_bailian20231229.models import ApplyFileUploadLeaseResponseBodyData, AddFileResponse
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


# todo 用户空间相关配置
access_key_id = "ACCESS_KEY_ID"
access_key_secret = "ACCESS_KEY_SECRET"
workspace_id = "WORKSPACE_ID"


def calculate_md5(file_path):
    """计算文档的 MD5 值。

    Args:
        file_path (str): 文档的路径。

    Returns:
        str: 文档的 MD5 值。
    """
    md5_hash = hashlib.md5()

    # 以二进制形式读取文件
    with open(file_path, "rb") as f:
        # 按块读取文件，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


class TmpFile:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> bailian20231229Client:
        """
        使用AK&SK初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考。
        # 建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html。
        config = open_api_models.Config(
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID。,
            access_key_id=access_key_id,
            # 必填，请确保代码运行环境设置了环境变量 ALIBABA_CLOUD_ACCESS_KEY_SECRET。,
            access_key_secret=access_key_secret
        )
        # Endpoint 请参考 https://api.aliyun.com/product/bailian
        config.endpoint = f'bailian.cn-beijing.aliyuncs.com'
        return bailian20231229Client(config)

    @staticmethod
    def apply_file(file_name, md_5) -> ApplyFileUploadLeaseResponseBodyData | None:
        client = TmpFile.create_client()
        apply_file_upload_lease_request = bailian_20231229_models.ApplyFileUploadLeaseRequest(
            file_name=file_name,
            md_5=md_5,
            size_in_bytes='111',
            category_type='SESSION_FILE'
        )
        runtime = util_models.RuntimeOptions()
        headers = {}
        try:
            # 复制代码运行请自行打印 API 的返回值
            response = client.apply_file_upload_lease_with_options('default', workspace_id,
                                                                   apply_file_upload_lease_request, headers, runtime)
            return response.body.data
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)

    @staticmethod
    def add_file(lease_id) -> AddFileResponse:
        client = TmpFile.create_client()
        add_file_request = bailian_20231229_models.AddFileRequest(
            lease_id=lease_id,
            parser='DASHSCOPE_DOCMIND',
            category_id='default',
            category_type='SESSION_FILE'
        )
        runtime = util_models.RuntimeOptions()
        headers = {}
        try:
            # 复制代码运行请自行打印 API 的返回值
            response = client.add_file_with_options(workspace_id, add_file_request, headers, runtime)
            return response
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


def upload_file(file_path):
    md_5 = calculate_md5(file_path)
    file_name = os.path.basename(file_path)
    data = TmpFile.apply_file(file_name, md_5)
    try:
        pre_signed_url = data.param.url
        # 设置请求头
        headers = {
            "X-bailian-extra": f"{data.param.headers['X-bailian-extra']}",
            "Content-Type": f"{data.param.headers['Content-Type']}"
        }

        # 读取文档并上传
        with open(file_path, 'rb') as file:
            # 下方设置请求方法用于文档上传，需与您在上一步中调用ApplyFileUploadLease接口实际返回的Data.Param中Method字段的值一致
            response = requests.put(pre_signed_url, data=file, headers=headers)
            # 检查响应状态码
            if response.status_code == 200:
                print("File uploaded successfully.")
            else:
                print(f"Failed to upload the file. ResponseCode: {response.status_code}")

        response = TmpFile.add_file(data.file_upload_lease_id)
        # 检查响应状态码
        if response.status_code == 200:
            print("File added successfully.")
            return response.body.data.file_id
        else:
            print(f"Failed to add the file. ResponseCode: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # 文档来源可以是本地，上传本地文档至百炼临时存储
    file_path = "./demo/test.png"
    upload_file(file_path)
