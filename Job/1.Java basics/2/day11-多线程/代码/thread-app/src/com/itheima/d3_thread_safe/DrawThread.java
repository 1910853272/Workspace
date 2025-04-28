<<<<<<< HEAD
package com.itheima.d3_thread_safe;

public class DrawThread extends Thread{
    private Account acc;
    public DrawThread(Account acc, String name){
        super(name);
        this.acc = acc;
    }
    @Override
    public void run() {
        // 取钱(小明，小红)
        acc.drawMoney(100000);
    }
}
=======
package com.itheima.d3_thread_safe;

public class DrawThread extends Thread{
    private Account acc;
    public DrawThread(Account acc, String name){
        super(name);
        this.acc = acc;
    }
    @Override
    public void run() {
        // 取钱(小明，小红)
        acc.drawMoney(100000);
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
