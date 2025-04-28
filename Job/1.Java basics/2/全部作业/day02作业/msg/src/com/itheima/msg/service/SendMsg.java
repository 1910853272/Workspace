package com.itheima.msg.service;

import com.itheima.msg.domain.CheckCode;

import java.util.ArrayList;
import java.util.Random;

public abstract class SendMsg {
    // 存储短信验证码的集合
    public ArrayList<CheckCode> arrayList = new ArrayList<>() ;
    private Random random = new Random();

    // 定义发送短信验证码时可选的数字字符
    public char[] chs = {'0' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9'} ;

    // 发送短信验证码的方法
    public void sendMsg(String phone) {
        // 补全代码
        // 1、产生随机验证码。
        String code = "";

        for (int i = 0; i < 4; i++) {
            code += chs[random.nextInt(chs.length)];
        }
        // 2、模拟发送短信
        send(phone, code);

        // 3、存储验证码
        CheckCode checkCode = new CheckCode();
        checkCode.setPhone(phone);
        checkCode.setCheckCode(code);
    }

    public abstract void send(String phone, String code);
}
