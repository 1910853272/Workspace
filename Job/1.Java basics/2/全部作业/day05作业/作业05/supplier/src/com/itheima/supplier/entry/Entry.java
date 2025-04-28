package com.itheima.supplier.entry;

import java.util.function.Supplier;

public class Entry {

    // 验证码可选字符
    private static final String OPTIONS_CAHS = "023456789abcdefghjklmnopqrstuvwxyz" ;

    public static void main(String[] args) {

        // 在等号的右边补全代码
        String verificationCode = null;

        // 输出
        System.out.println("本次生成的验证码为：" + verificationCode);

    }

    // 调用该方法生成一个4位的随机验证码
    public static String verificationCode(Supplier<String> supplier) {
        return supplier.get() ;
    }

}
