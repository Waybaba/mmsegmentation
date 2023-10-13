""" Convert Python configuration files to YAML format.


# How to Convert

1. use convert_cfg.py to generate basic
2. cp to configs_hydra/

cp -rv debug/configs_hydra/{pspnet,segformer,segmenter,deeplabv3,deeplabv3plus} \
	configs_hydra/experiment/
cp -rv debug/configs_hydra/_base_ configs_hydra/
2. handle _delete_ keys
3. change default keys to key wised


"""

import os
import yaml
import copy
from pathlib import Path
import pyrootutils

SRC_CONFIGS = 'configs'
DST_CONFIGS = 'debug/configs_hydra'

def represent_list_indent(dumper, data):
    """Make sure to indent lists."""
    return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=False, indent_offset=2)

def convert_global_vars_to_dict(global_vars):
	"""Convert global variables from exec to a filtered dictionary."""
	return {k: copy.deepcopy(v) for k, v in global_vars.items() if not k.startswith("__")}

def numeric_list_representer(dumper, data):
    """Represent lists of numbers (integers and floats) in a compact form."""
    if all(isinstance(item, (int, float)) for item in data):
        return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_list(data)


from collections import OrderedDict

def order_dict(data):
    """Ensure 'defaults' is the first key in the dictionary if it exists."""
    if "defaults" not in data:
        return data

    ordered_data = OrderedDict()
    ordered_data['defaults'] = data.pop('defaults')
    for key, value in data.items():
        ordered_data[key] = value
    return ordered_data

def dereference_data(data):
	"""Recursively create deep copies of shared references."""
	if isinstance(data, list):
		return [dereference_data(item) for item in data]
	if isinstance(data, dict):
		return {k: dereference_data(v) for k, v in data.items()}
	return copy.deepcopy(data)

def tuple_representer(dumper, data):
	"""Represent tuple in a custom way for YAML."""
	return dumper.represent_dict({'_target_': 'builtins.tuple', '_args_': [list(data)]})

def convert_py_to_yaml(py_file_path):
	"""Convert a Python configuration file to a YAML format."""
	with open(py_file_path, 'r') as f:
		content = f.read()

	global_vars = {}
	try:
		exec(content, global_vars)
	except Exception as e:
		raise ValueError(f"Error executing the Python file: {e}")

	global_vars = convert_global_vars_to_dict(global_vars)

	yaml_data = {key: value for key, value in global_vars.items() if key != "_base_"}

	if "_base_" in global_vars:
		base_values = global_vars["_base_"]
		if isinstance(base_values, str): # single base
			base_values = [base_values]
		yaml_data["defaults"] = [
			item.replace('.py', '.yaml').replace('../', '/') \
				.replace('/_base_/models/', '/_base_/models: ') \
				.replace('/_base_/datasets/', '/_base_/datasets: ') \
				.replace('/_base_/default_runtime', '/_base_/default_runtime') \
				.replace('/_base_/schedules/', '/_base_/schedules: ') \
				for item in base_values
			]
		# to dict
		for i in range(len(yaml_data["defaults"])):
			if ': ' in yaml_data["defaults"][i]:
				yaml_data["defaults"][i] = {yaml_data["defaults"][i].split(': ')[0]: yaml_data["defaults"][i].split(': ')[1]}

	yaml_data = dereference_data(yaml_data)
	yaml.add_representer(tuple, tuple_representer)
	yaml.add_representer(list, numeric_list_representer)  # <-- Add this line
	# yaml.add_representer(list, represent_list_indent)

	def ordered_dict_representer(dumper, data):
		return dumper.represent_dict(data.items())

	yaml.add_representer(OrderedDict, ordered_dict_representer)
	return yaml.dump(order_dict(yaml_data), default_flow_style=False)

def main():
	root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
	src_directory = root / SRC_CONFIGS
	dst_directory = root / DST_CONFIGS
	final_directory = root / "configs_hydra"
	dst_directory.mkdir(parents=True, exist_ok=True)
	final_directory.mkdir(parents=True, exist_ok=True)


	for root_, dirs, files in os.walk(src_directory):
		dst_subdir = Path(root_.replace(str(src_directory), str(dst_directory)))
		dst_subdir.mkdir(parents=True, exist_ok=True)

		for filename in files:
			src_file = Path(root_) / filename

			if filename.endswith(".py"):
				yaml_data = convert_py_to_yaml(src_file)
				with (dst_subdir / (src_file.stem + '.yaml')).open('w') as f:
					f.write("# @package _global_\n")
					f.write(yaml_data)
			else:
				with open(src_file, 'r') as src_f, (dst_subdir / filename).open('w') as dst_f:
					dst_f.write(src_f.read())
			
			print(f"Converted {dst_subdir / filename}")

if __name__ == "__main__":
	main()
