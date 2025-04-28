<<<<<<< HEAD
package test04;

public class ProductItem {
    //订单项编号
    private int productItemNumber;
    //商品名称
    private String productName;
    //商品价格
    private double price;

    public ProductItem() {
    }

    public ProductItem(int productItemNumber, String productName, double price) {
        this.productItemNumber = productItemNumber;
        this.productName = productName;
        this.price = price;
    }

    public int getProductItemNumber() {
        return productItemNumber;
    }

    public void setProductItemNumber(int productItemNumber) {
        this.productItemNumber = productItemNumber;
    }

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    @Override
    public String toString() {
        return "ProductItem{" +
                "productItemNumber=" + productItemNumber +
                ", productName='" + productName + '\'' +
                ", price=" + price +
                '}';
    }
}
=======
package test04;

public class ProductItem {
    //订单项编号
    private int productItemNumber;
    //商品名称
    private String productName;
    //商品价格
    private double price;

    public ProductItem() {
    }

    public ProductItem(int productItemNumber, String productName, double price) {
        this.productItemNumber = productItemNumber;
        this.productName = productName;
        this.price = price;
    }

    public int getProductItemNumber() {
        return productItemNumber;
    }

    public void setProductItemNumber(int productItemNumber) {
        this.productItemNumber = productItemNumber;
    }

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    @Override
    public String toString() {
        return "ProductItem{" +
                "productItemNumber=" + productItemNumber +
                ", productName='" + productName + '\'' +
                ", price=" + price +
                '}';
    }
}
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
