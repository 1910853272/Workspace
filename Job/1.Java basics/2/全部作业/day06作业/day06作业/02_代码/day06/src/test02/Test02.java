package test02;

import java.util.ArrayList;

public class Test02 {
    public static void main(String[] args) {
        //1.定义一个集合,存储学生对象
        ArrayList<Student> list = new ArrayList<>();
        //2.创建4个对象,并赋值,存储到集合中
        list.add(new Student(1,"张无忌","河北"));
        list.add(new Student(2,"张三丰","河南"));
        list.add(new Student(3,"赵敏","蒙古"));
        list.add(new Student(4,"周芷若","四川"));
        //3.遍历集合,判断名字如果是"张三丰",重新将adress赋值成"山东"
        for (int i = 0; i < list.size(); i++) {
            Student student = list.get(i);
            String name = student.getName();
            if ("张三丰".equals(name)){
                Student s = new Student(student.getId(), name, "山东");
                //4.调用set方法,替换
                list.set(i,s);
            }
        }
        for (int i = 0; i < list.size(); i++) {
            System.out.println(list.get(i).getId()+"..."+list.get(i).getName()+"..."+list.get(i).getAdress());
        }

        //5.遍历
    }
}
