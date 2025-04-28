package test02;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

public class Test {
    public static void main(String[] args) {
//        1.创建Random随机数对象
        Random random = new Random();
//        2.创建HashSet用于保存不重复的红球
        HashSet<Integer> hashSet = new HashSet<>();

//        3.循环判断红球数量是否小于6个
        while(hashSet.size()<6){
            //4.球数量小于6个就产生一个红球.添加到HashSet中
            int num = random.nextInt(33) + 1;
            hashSet.add(num);
        }
//        5.创建ArrayList集合
        ArrayList<Integer> list = new ArrayList<>();
        for (Integer element : hashSet) {
            list.add(element);
        }
//        6.排序
        Collections.sort(list);
//        7.再生成一个蓝球
        int blue = random.nextInt(13) + 1;
//        8.打印中奖号码
        System.out.println("红色号为:");
        for (Integer red : list) {
            System.out.print(red+" ");
        }
        System.out.println();
        System.out.println("蓝色号为:");
        System.out.println(blue);
    }
}
