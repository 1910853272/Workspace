<<<<<<< HEAD
package com.itheima.msg.service;


public class TencentSendMsg extends SendMsg{

    @Override
    public void send(String phone, String code) {
        System.out.println("腾讯云：" + phone + "发送了验证码" + code);
    }

}
=======
package com.itheima.msg.service;


public class TencentSendMsg extends SendMsg{

    @Override
    public void send(String phone, String code) {
        System.out.println("腾讯云：" + phone + "发送了验证码" + code);
    }

}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
