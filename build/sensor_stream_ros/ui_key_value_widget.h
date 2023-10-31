/********************************************************************************
** Form generated from reading UI file 'key_value_widget.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_KEY_VALUE_WIDGET_H
#define UI_KEY_VALUE_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_KeyValueWidget
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout;
    QLabel *key;
    QHBoxLayout *horizontalLayout;
    QDoubleSpinBox *value;

    void setupUi(QWidget *KeyValueWidget)
    {
        if (KeyValueWidget->objectName().isEmpty())
            KeyValueWidget->setObjectName(QString::fromUtf8("KeyValueWidget"));
        KeyValueWidget->resize(279, 48);
        verticalLayout = new QVBoxLayout(KeyValueWidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        key = new QLabel(KeyValueWidget);
        key->setObjectName(QString::fromUtf8("key"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(key->sizePolicy().hasHeightForWidth());
        key->setSizePolicy(sizePolicy);
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        key->setFont(font);

        formLayout->setWidget(0, QFormLayout::LabelRole, key);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        value = new QDoubleSpinBox(KeyValueWidget);
        value->setObjectName(QString::fromUtf8("value"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(value->sizePolicy().hasHeightForWidth());
        value->setSizePolicy(sizePolicy1);
        value->setDecimals(4);
        value->setMinimum(-99999999.000000000000000);
        value->setMaximum(99999999.000000000000000);

        horizontalLayout->addWidget(value);


        formLayout->setLayout(0, QFormLayout::FieldRole, horizontalLayout);


        verticalLayout->addLayout(formLayout);


        retranslateUi(KeyValueWidget);

        QMetaObject::connectSlotsByName(KeyValueWidget);
    } // setupUi

    void retranslateUi(QWidget *KeyValueWidget)
    {
        KeyValueWidget->setWindowTitle(QApplication::translate("KeyValueWidget", "Form", nullptr));
        key->setText(QApplication::translate("KeyValueWidget", "key", nullptr));
    } // retranslateUi

};

namespace Ui {
    class KeyValueWidget: public Ui_KeyValueWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_KEY_VALUE_WIDGET_H
