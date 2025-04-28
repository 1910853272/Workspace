<<<<<<< HEAD
package com.itheima.d11_interface_attention;

public class Test {
    public static void main(String[] args) {
        // 目标：理解接口的多继承。
    }
}

interface A{
    void test1();
}
interface B{
    void test2();
}
interface C{}

// 接口是多继承的
interface D extends C, B, A{

}

class E implements D{
    @Override
    public void test1() {

    }

    @Override
    public void test2() {

    }
}


=======
package com.itheima.d11_interface_attention;

public class Test {
    public static void main(String[] args) {
        // 目标：理解接口的多继承。
    }
}

interface A{
    void test1();
}
interface B{
    void test2();
}
interface C{}

// 接口是多继承的
interface D extends C, B, A{

}

class E implements D{
    @Override
    public void test1() {

    }

    @Override
    public void test2() {

    }
}


>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
