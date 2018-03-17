//#include "keras_model.h"
//
//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\core\core.hpp>
//#include <iostream>
//
//using namespace std;
//using namespace keras;
//
//string labels[10] = {"双手平举","弯腰","行走","半蹲","单手挥手","侧身舒展","叉腰","趴地","打电话","两人交流"};
//int main(int argc, char *argv[]) 
//{
//  if(argc != 2) 
//  {
//    cout << "Wrong input, going to exit." << endl;
//    cout << "There should be arguments: dumped_cnn_file input_sample output_file." << endl;
//    return -1;
//  }
//  string dumped_cnn = "cnn1.dumped";
//  string input_data = argv[1];
//
//  cout << "Testing network from " << dumped_cnn << " on data from " << input_data << endl;
//
//  cout << "read cnn" << endl;
//  KerasModel m(dumped_cnn, true);
//
//
//  DataChunk *sample = new DataChunk2D();
//
//  cout <<endl<< "read data" << endl;
//  //sample->read_from_file(input_data);
//  Mat image = imread(input_data+".bmp");
//  resize(image, image, Size(224, 224));
//  sample->read_from_image(image);
//
//  cout <<endl<< "compute" << endl;
//  std::vector<float> response = m.compute_output(sample);
//  delete sample;
//
//  cout << endl<< "result:" << endl;
//  ofstream fout(input_data+".dat");
//  int resultLabel=0;
//  float maxResult = 0.0;
//  for(unsigned int i = 0; i < response.size(); i++) 
//  {
//	  cout << labels[i] << ": " << response[i] << endl;
//	  if (maxResult < response[i])
//	  {
//		  maxResult = response[i];
//		  resultLabel = i;
//	  }
//	  fout << response[i] << " ";
//  }
//  cout << "当前动作为: " << labels[resultLabel] << endl;
//  fout << labels[resultLabel] << " ";
//  fout.close();
//  cout << "finish" << endl;
//  return 0;
//}
