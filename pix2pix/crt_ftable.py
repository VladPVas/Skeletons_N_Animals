import glob, os, sys
import numpy as np

OUT_FOLDER = './dataset' 
INPUT_IMG_FOLDER = './dataset/skeletons' 
OUTPUT_IMG_FOLDER = './dataset/animal' 


input_img_files = np.array(glob.glob(INPUT_IMG_FOLDER+'/*.jpg')+
                           glob.glob(INPUT_IMG_FOLDER+'/*.jpeg')+
                           glob.glob(INPUT_IMG_FOLDER+'/*.png'), dtype=np.object)
output_img_files = np.array(glob.glob(OUTPUT_IMG_FOLDER+'/*.jpg')+
                            glob.glob(OUTPUT_IMG_FOLDER+'/*.jpeg')+
                            glob.glob(OUTPUT_IMG_FOLDER+'/*.png'), dtype=np.object)


train_t = []
for input_img_file in input_img_files:
    inp = input_img_file
    
    out = None
    for output_img_file in output_img_files:
        if os.path.basename(input_img_file).split('.')[0]==os.path.basename(output_img_file).split('.')[0]:
            out = output_img_file
    if out is None:
        sys.exit("Не найден файл-пара для файла {}! Проверьте данные".format(inp))
    train_t.append([inp,out])

train_t = np.array(train_t)

np.savetxt(OUT_FOLDER+'/train.txt', train_t,fmt='%s',delimiter='\t')
