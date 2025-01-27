-- In this file, we will create the database and tables for the delivery management system. 
-- The tables include Customers, Drivers, Packages, Routes, Orders, and TrackingLogs.
-- There are some column names are commented out due to the availibity of dataset.



-- Create the DeliveryDB database if it doesn't exist
DROP DATABASE DeliveryDB;
CREATE DATABASE DeliveryDB;
USE DeliveryDB;


-- Table for storing customer details
CREATE TABLE `DeliveryDB`.`Customers` (
    CustomerID INT PRIMARY KEY AUTO_INCREMENT,
    -- Name VARCHAR(100) NOT NULL,
    -- Address VARCHAR(255) NOT NULL,
    Latitude DECIMAL(10, 8) NOT NULL,
    Longtitude DECIMAL(11, 8) NOT NULL,
    City VARCHAR(50)
    -- State VARCHAR(50)
);

-- Table for storing delivery driver details
CREATE TABLE `DeliveryDB`.`Drivers` (
    DriverID INT PRIMARY KEY AUTO_INCREMENT,
    -- Name VARCHAR(100) NOT NULL,
    -- PhoneNumber VARCHAR(20),
    -- VehicleID VARCHAR(20),
    -- CurrentLocation POINT, -- For storing GPS coordinates
    Status ENUM('Available', 'On Delivery', 'Off Duty') DEFAULT 'Available'
);

-- Table for storing delivery package details
CREATE TABLE `DeliveryDB`.`Packages`(
    PackageID INT PRIMARY KEY AUTO_INCREMENT,
    OrderID VARCHAR(50) NOT NULL,
    -- Weight DECIMAL(10, 2) NOT NULL,
    -- Dimensions VARCHAR(50), -- e.g., "10x10x10"
    DeliveryStatus ENUM('Pending', 'In Transit', 'Delivered', 'Returned') DEFAULT 'Pending',
    EstimatedDeliveryTime DATETIME,
    ActualDeliveryTime DATETIME,
    CustomerID INT,
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Table for storing delivery route details
CREATE TABLE `DeliveryDB`.`Routes` (
    RouteID INT PRIMARY KEY AUTO_INCREMENT,
    DriverID INT,
    StartLocation POINT, -- GPS coordinates
    EndLocation POINT, -- GPS coordinates
    Distance DECIMAL(10, 2), -- In kilometers or miles
    RouteStatus ENUM('Planned', 'In Progress', 'Completed', 'Cancelled') DEFAULT 'Planned',
    StartTime DATETIME,
    EndTime DATETIME,
    FOREIGN KEY (DriverID) REFERENCES Drivers(DriverID)
);

-- Table for storing order details
CREATE TABLE `DeliveryDB`.`Orders` (
    OrderID VARCHAR(50) PRIMARY KEY,
    CustomerID INT,
    OrderDate DATETIME NOT NULL,
    DeliveryDate DATETIME,
    -- TotalAmount DECIMAL(10, 2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Table for logging delivery tracking events
CREATE TABLE `DeliveryDB`.`TrackingLogs` (
    LogID INT PRIMARY KEY AUTO_INCREMENT,
    PackageID INT,
    EventDescription VARCHAR(255),
    EventTime DATETIME,
    Location POINT, -- GPS coordinates
    FOREIGN KEY (PackageID) REFERENCES Packages(PackageID)
);
