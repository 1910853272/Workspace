package com.itheima.atm.service.impl;

import com.itheima.atm.service.AtmService;

public class AtmServiceImpl implements AtmService {          // atm机业务处理类

    // 定义总金额
    private double totalMoney = 20000 ;

    // 取钱的方法
    @Override
    public double drawMoney(double money) {
        return totalMoney - money ;
    }

}
