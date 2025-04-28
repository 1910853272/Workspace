<<<<<<< HEAD
package com.itheima.d4_proxy;

public class BigStar implements Star{
    private String name;

    public BigStar(String name) {
        this.name = name;
    }

    public String sing(String name){
        System.out.println(this.name + "正在唱：" + name);
        return "谢谢！谢谢！";
    }

    public void dance(){
        System.out.println(this.name  + "正在优美的跳舞~~");
    }
}
=======
package com.itheima.d4_proxy;

public class BigStar implements Star{
    private String name;

    public BigStar(String name) {
        this.name = name;
    }

    public String sing(String name){
        System.out.println(this.name + "正在唱：" + name);
        return "谢谢！谢谢！";
    }

    public void dance(){
        System.out.println(this.name  + "正在优美的跳舞~~");
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
