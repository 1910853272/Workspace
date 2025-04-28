package test04;

import java.util.ArrayList;
import java.util.Random;

public class Test04 {
    public static void main(String[] args) {
        //创建一个存储订单项的订单集合
        ArrayList<ProductItem> productList = new ArrayList<>();
        //创建一个存储订单的集合
        ArrayList<ArrayList<ProductItem>> list = new ArrayList<>();

        //将订单项存储到订单集合中
        productList.add(new ProductItem(new Random().nextInt(1000000),"键盘",317));
        productList.add(new ProductItem(new Random().nextInt(1000000),"漱口水",157));
        productList.add(new ProductItem(new Random().nextInt(1000000),"U盘",118));
        productList.add(new ProductItem(new Random().nextInt(1000000),"耳机",298));

        //将订单集合存储到大集合中
        list.add(productList);

        //遍历
        for (ArrayList<ProductItem> productItemArrayList : list) {
            for (ProductItem productItem : productItemArrayList) {
                System.out.println(productItem.getProductItemNumber()+"..."+productItem.getProductName()+"..."+productItem.getPrice());
            }
        }
    }
}
