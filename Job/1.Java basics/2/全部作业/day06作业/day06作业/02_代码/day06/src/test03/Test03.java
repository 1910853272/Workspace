package test03;

import java.util.ArrayList;
import java.util.Scanner;

public class Test03 {
    public static void main(String[] args) {
        //1.键盘录入,Scanner
        Scanner sc = new Scanner(System.in);
        //2.创建集合
        ArrayList<User> list = new ArrayList<>();
        //3.准备点集合中有的数据
        User u1 = new User("liuyan", "123");
        User u2 = new User("mingri", "456");
        //4.将准备数据添加到集合中
        list.add(u1);
        list.add(u2);

        while(true){
            //5.键盘录入要注册的用户名以及密码
            System.out.println("请你输入要注册的用户名:");
            String name = sc.next();//liuyan
            boolean b = isUser(name, list);
            if (b){
                System.out.println("对不起,您注册的用户名已经存在!");
            }else{
                System.out.println("请你输入要注册的密码:");
                String pwd = sc.next();
                //6.将name和pwd封装成User对象,存到集合中
                User user = new User(name, pwd);
                list.add(user);

                System.out.println("注册成功!");
                //调用展示用户名的方法
                showUser(list);
                break;
            }

        }


    }

    private static void showUser(ArrayList<User> list) {
        for (User user : list) {
            System.out.println(user.getUsername()+"..."+user.getPassword());
        }
    }

    private static boolean isUser(String name, ArrayList<User> list) {//list:liuyan  mingri    name:liuyan
        //定义一个标记,表示集合中没有要注册的用户
        boolean flag = false;

        //遍历集合
        for (int i = 0; i < list.size(); i++) {
            User user = list.get(i);
            if (name.equals(user.getUsername())){
                //证明集合中有要注册的用户名
                flag = true;
                break;
            }
        }
        return flag;
    }
}
