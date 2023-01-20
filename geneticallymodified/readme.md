### track information

遺伝子組み換えこいしﾁｬﾝ  
genetically modified Koichi chan  
https://www.youtube.com/watch?v=t4i23nLpr8s  

original: ハルトマンの妖怪少女 by ZUN  
arranged by laplace (@yamadala)  

### how to make .wav file

1. install libraries as required,  
2. copy script (geneticallymodified.py) to ipython notebook,  
3. run all cells,  
4. download .wav file from a audio control,  
  or export by IPython.lib.display.Audio object,  

### comments

* programmed by laplace in Python,  
* using genetic algorithm,  
  * generate 4 bars in each generation,  
  * exchange notes in different bars by crossover,  
  * rewrite notes in each bar by mutation,  
    * mutation method is rotation, inversion or overwriting,  
  * at least one of crossover or mutation is performed,  
