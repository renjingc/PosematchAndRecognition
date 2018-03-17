#ifndef __FILES_TOOL_H
#define __FILES_TOOL_H

#include <iostream>
#include <string>
#include <vector>
#include <direct.h>
#include <io.h>
#include <stdio.h>
#include <fstream>

using namespace std;

void getFilesAllName(string path, vector<string>& files);//读取某给定路径下所有文件名，不包括路径
void getAllFiles(string path, vector<string>& files);//读取某给定路径下所有文件夹与文件名称，并带完整路径
void getJustCurrentDir(string path, vector<string>& files);//只读取某给定路径下的当前文件夹名
void getJustCurrentFile(string path, vector<string>& files);//只读取某给定路径下的当前文件名
void getFilesAll(string path, vector<string>& files);//只读取某给定路径下的所有文件名(即包含当前目录及子目录的文件)

#endif