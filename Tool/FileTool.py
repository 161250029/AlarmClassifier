import os
class FileTool:
    # 根据后缀名筛选文件名
    def get_file_by_ext(self , dir_path , ext = None):
        allfiles = []
        needExtFilter = (ext != None)
        for root , dirs , files in os.walk(dir_path):
            for file_name in files:
                filepath = os.path.join(root, file_name)
                extension = os.path.splitext(filepath)[1][1:]
                if needExtFilter and extension in ext:
                    allfiles.append(filepath)
                elif not needExtFilter:
                    allfiles.append(filepath)
        return allfiles

if __name__ == '__main__':
    print(FileTool().get_file_by_ext('/root/GY/Experiment/camel-core/2.22.3/' , ['pkl']))
