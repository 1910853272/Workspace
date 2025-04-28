<<<<<<< HEAD
package com.itheima.d3_inner_class3;

public class Test2 {
    public static void main(String[] args) {
        // 目标：掌握匿名的常见使用场景。
//        Swimming s1 = new Swimming(){
//            @Override
//            public void swim() {
//                System.out.println("狗🏊‍飞快~~~~");
//            }
//        };
//        go(s1);

        go(new Swimming(){
            @Override
            public void swim() {
                System.out.println("狗🏊‍飞快~~~~");
            }
        });

    }

    // 设计一个方法，可以接收swimming接口的一切实现类对象进来参加游泳比赛。
    public static void go(Swimming s){
        System.out.println("开始-----------------------");
        s.swim();
    }
}



// 猫和狗都要参加游泳比赛
interface Swimming{
    void swim();
}
=======
package com.itheima.d3_inner_class3;

public class Test2 {
    public static void main(String[] args) {
        // 目标：掌握匿名的常见使用场景。
//        Swimming s1 = new Swimming(){
//            @Override
//            public void swim() {
//                System.out.println("狗🏊‍飞快~~~~");
//            }
//        };
//        go(s1);

        go(new Swimming(){
            @Override
            public void swim() {
                System.out.println("狗🏊‍飞快~~~~");
            }
        });

    }

    // 设计一个方法，可以接收swimming接口的一切实现类对象进来参加游泳比赛。
    public static void go(Swimming s){
        System.out.println("开始-----------------------");
        s.swim();
    }
}



// 猫和狗都要参加游泳比赛
interface Swimming{
    void swim();
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
