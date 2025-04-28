package com.itheima.msg.domain;

public class CheckCode {            // 记录短信验证码的类

    private String phone ;          // 手机号码
    private String checkCode ;      // 验证码

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }

    public String getCheckCode() {
        return checkCode;
    }

    public void setCheckCode(String checkCode) {
        this.checkCode = checkCode;
    }
}
