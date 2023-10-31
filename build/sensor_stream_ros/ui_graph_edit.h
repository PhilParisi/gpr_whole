/********************************************************************************
** Form generated from reading UI file 'graph_edit.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GRAPH_EDIT_H
#define UI_GRAPH_EDIT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GraphEdit
{
public:
    QVBoxLayout *verticalLayout;
    QFormLayout *formLayout;
    QLabel *key_lbl;
    QComboBox *key;
    QLabel *label;
    QComboBox *line_style;
    QPushButton *color_btn;
    QPushButton *remove;

    void setupUi(QWidget *GraphEdit)
    {
        if (GraphEdit->objectName().isEmpty())
            GraphEdit->setObjectName(QString::fromUtf8("GraphEdit"));
        GraphEdit->resize(400, 300);
        verticalLayout = new QVBoxLayout(GraphEdit);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        formLayout = new QFormLayout();
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        key_lbl = new QLabel(GraphEdit);
        key_lbl->setObjectName(QString::fromUtf8("key_lbl"));

        formLayout->setWidget(0, QFormLayout::LabelRole, key_lbl);

        key = new QComboBox(GraphEdit);
        key->setObjectName(QString::fromUtf8("key"));

        formLayout->setWidget(0, QFormLayout::FieldRole, key);

        label = new QLabel(GraphEdit);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label);

        line_style = new QComboBox(GraphEdit);
        line_style->setObjectName(QString::fromUtf8("line_style"));

        formLayout->setWidget(1, QFormLayout::FieldRole, line_style);


        verticalLayout->addLayout(formLayout);

        color_btn = new QPushButton(GraphEdit);
        color_btn->setObjectName(QString::fromUtf8("color_btn"));

        verticalLayout->addWidget(color_btn);

        remove = new QPushButton(GraphEdit);
        remove->setObjectName(QString::fromUtf8("remove"));

        verticalLayout->addWidget(remove);


        retranslateUi(GraphEdit);

        QMetaObject::connectSlotsByName(GraphEdit);
    } // setupUi

    void retranslateUi(QWidget *GraphEdit)
    {
        GraphEdit->setWindowTitle(QApplication::translate("GraphEdit", "Form", nullptr));
        key_lbl->setText(QApplication::translate("GraphEdit", "Key:", nullptr));
        label->setText(QApplication::translate("GraphEdit", "Line Style", nullptr));
        color_btn->setText(QApplication::translate("GraphEdit", "Set Color...", nullptr));
        remove->setText(QApplication::translate("GraphEdit", "Remove Graph", nullptr));
    } // retranslateUi

};

namespace Ui {
    class GraphEdit: public Ui_GraphEdit {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GRAPH_EDIT_H
