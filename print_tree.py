import os

def print_tree(root, indent="", max_files=5):
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return

    files_shown = 0
    for i, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = (i == len(items) - 1 or files_shown == max_files)
        connector = "└── " if is_last else "├── "
        print(indent + connector + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, indent + extension, max_files=max_files)
        else:
            files_shown += 1
            if files_shown == max_files:
                print(indent + "│   ...")
                break

# 设置你的项目根目录
root_dir = "./DUT-Anti-UAV"  # 比如 ./data
print(f"📁 目录结构（最多每个文件夹显示 5 个文件）：{os.path.abspath(root_dir)}")
print_tree(root_dir, max_files=5)