package test01;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class Test {
    public static void main(String[] args) {
        //1.创建集合
        ArrayList<Student> list = new ArrayList<>();
        //2.将4个学生对象,存储到集合中
        list.add(new Student(1,"小亮",99));
        list.add(new Student(2,"小勇",85));
        list.add(new Student(3,"小响",90));
        list.add(new Student(4,"小强",78));

        //3.排序
        Collections.sort(list, new Comparator<Student>() {
            @Override
            public int compare(Student o1, Student o2) {
                return o1.getScore()-o2.getScore();
            }
        });

        //4.遍历集合
        for (int i = 0; i < list.size(); i++) {
            Student student = list.get(i);
            System.out.println(student.getId()+"..."+student.getName()+"..."+student.getScore());
        }
    }
}
