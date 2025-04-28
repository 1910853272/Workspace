package com.itheima.msg.service;


public class TencentSendMsg extends SendMsg{

    @Override
    public void send(String phone, String code) {
        System.out.println("腾讯云：" + phone + "发送了验证码" + code);
    }

}
