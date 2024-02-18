// MIT License
//
// Copyright (c) 2018 Takafumi Horiuchi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Source: https://github.com/takafumihoriuchi/MNIST_for_C

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define MNIST_TRAIN_IMAGE_PATH "./data/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL_PATH "./data/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGE_PATH "./data/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL_PATH "./data/t10k-labels-idx1-ubyte"

#define MNIST_VEC_SIZE 784 // 28*28
#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[MNIST_NUM_TRAIN][MNIST_VEC_SIZE];
unsigned char test_image_char[MNIST_NUM_TEST][MNIST_VEC_SIZE];
unsigned char train_label_char[MNIST_NUM_TRAIN][1];
unsigned char test_label_char[MNIST_NUM_TEST][1];

double train_image[MNIST_NUM_TRAIN][MNIST_VEC_SIZE];
double test_image[MNIST_NUM_TEST][MNIST_VEC_SIZE];
int train_label[MNIST_NUM_TRAIN];
int test_label[MNIST_NUM_TEST];

void flip_long(unsigned char* ptr)
{
    register unsigned char val;
    
    // Swap 1st and 4th bytes.
    val = *(ptr);
    *(ptr) = *(ptr + 3);
    *(ptr+3) = val;
    
    // Swap 2nd and 3rd bytes.
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr + 1);
    *(ptr+1) = val;
}

void read_mnist_char(
    char *file_path, int num_data, int len_info, int arr_n, 
    unsigned char data_char[][arr_n], int info_arr[]
){
    int i, j, k, fd;
    unsigned char *ptr;

    if((fd = open(file_path, O_RDONLY)) == -1) 
    {
        fprintf(stderr, "Couldn't open image file");
        exit(-1);
    }
    
    read(fd, info_arr, len_info * sizeof(int));
    
    // Read information about size of data.
    for(i = 0; i < len_info; i++) 
    { 
        ptr = (unsigned char*)(info_arr + i);
        flip_long(ptr);
        ptr = ptr + sizeof(int);
    }
    
    // Read mnist numbers (pixels|labels).
    for(i=0; i < num_data; i++) 
    {
        read(fd, data_char[i], arr_n * sizeof(unsigned char));   
    }

    close(fd);
}

void image_char2double(
    int num_data, 
    unsigned char data_image_char[][MNIST_VEC_SIZE], 
    double data_image[][MNIST_VEC_SIZE]
){
    int i, j;
    for(i = 0; i < num_data; i++)
    {
        for(j=0; j < MNIST_VEC_SIZE; j++)
        {
            data_image[i][j]  = (double)data_image_char[i][j] / 255.0;
        }
    }
}

void label_char2int(int num_data, unsigned char data_label_char[][1], int data_label[])
{
    int i;
    for(i=0; i<num_data; i++)
    {
        data_label[i] = (int)data_label_char[i][0];
    }
}

void load_mnist()
{
    read_mnist_char(MNIST_TRAIN_IMAGE_PATH, MNIST_NUM_TRAIN, LEN_INFO_IMAGE, MNIST_VEC_SIZE, train_image_char, info_image);
    image_char2double(MNIST_NUM_TRAIN, train_image_char, train_image);

    read_mnist_char(MNIST_TEST_IMAGE_PATH, MNIST_NUM_TEST, LEN_INFO_IMAGE, MNIST_VEC_SIZE, test_image_char, info_image);
    image_char2double(MNIST_NUM_TEST, test_image_char, test_image);
    
    read_mnist_char(MNIST_TRAIN_LABEL_PATH, MNIST_NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    label_char2int(MNIST_NUM_TRAIN, train_label_char, train_label);
    
    read_mnist_char(MNIST_TEST_LABEL_PATH, MNIST_NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    label_char2int(MNIST_NUM_TEST, test_label_char, test_label);
}
