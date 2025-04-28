package test01;

import java.util.ArrayList;

public class Test01 {
    public static void main(String[] args) {
//        1.定义一个学生类,包含两个私有的属性,构造方法,以及get/set方法
//        2.定义一个ArrayList集合 list
        ArrayList<Student> list = new ArrayList<>();
//        3.创建4个学生对象,调用add方法将4个方法存到list集合中
        list.add(new Student(1,"小亮",99));
        list.add(new Student(2,"小强",85));
        list.add(new Student(3,"小响",90));
        list.add(new Student(4,"小勇",75));

//        4.定义一个变量score,来接收集合中第一个学生的成绩
        int score = list.get(0).getScore();
//        5.定义一个变量index,接收最低成绩学生的在集合中出现的索引
        int index = 0;
         /* 6.遍历集合,获取出来每一个学生对象的成绩,去和第一个学生成绩做比较
            如果小于第一个学生成绩,就将成绩低的赋值给score,然后将对应的索引赋值给index*/
        for (int i = 0; i < list.size(); i++) {
            int chengji = list.get(i).getScore();
            if (score>chengji){
                index = i;
            }
        }
//        7.根据index将对应的学生调用remove删除
        list.remove(index);
//        8.遍历集合中学生的学号,姓名,考试成绩

        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i).getId()+"..."+list.get(i).getName()+list.get(i).getScore());
        }
    }
}
