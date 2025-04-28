<<<<<<< HEAD
package test03;

import java.util.HashSet;
import java.util.Scanner;

public class Test03 {
    public static void main(String[] args) {
     //1.创建Set集合
        HashSet<String> hashSet = new HashSet<>();
        //2.创建Scanner对象
        Scanner sc = new Scanner(System.in);
        while(true){
            System.out.println("请你输入要注册的用户名:");
            String name = sc.next();
            //3.存储
            boolean result = hashSet.add(name);
            if (result){
                System.out.println("注册成功");
            }else{
                System.out.println("注册失败");
            }
        }
    }
}
=======
package test03;

import java.util.HashSet;
import java.util.Scanner;

public class Test03 {
    public static void main(String[] args) {
     //1.创建Set集合
        HashSet<String> hashSet = new HashSet<>();
        //2.创建Scanner对象
        Scanner sc = new Scanner(System.in);
        while(true){
            System.out.println("请你输入要注册的用户名:");
            String name = sc.next();
            //3.存储
            boolean result = hashSet.add(name);
            if (result){
                System.out.println("注册成功");
            }else{
                System.out.println("注册失败");
            }
        }
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
