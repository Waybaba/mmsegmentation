import os
import yaml
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])


def convert_py_to_yaml(py_file_path):
    """
    return content of yaml file
    """
    return "template"

def main():
    src_directory = root / 'configs'
    dst_directory = root / 'hydra_configs'

    # 如果目标目录不存在，则创建
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)

    # 遍历源目录
    for root, dirs, files in os.walk(src_directory):
        # 对于每一个目录，创建相应的目标目录
        for dir_name in dirs:
            src_subdir = os.path.join(root, dir_name)
            dst_subdir = src_subdir.replace(src_directory, dst_directory)
            
            if not os.path.exists(dst_subdir):
                os.makedirs(dst_subdir)

        # 对于每一个.py文件，转换为.yaml文件并保存到目标目录
        for filename in files:
            if filename.endswith(".py"):
                src_file = os.path.join(root, filename)
                yaml_data = convert_py_to_yaml(src_file)
                
                yaml_filename = os.path.splitext(filename)[0] + '.yaml'
                dst_file = os.path.join(root.replace(src_directory, dst_directory), yaml_filename)
                
                with open(dst_file, 'w') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)

if __name__ == "__main__":
    main()
