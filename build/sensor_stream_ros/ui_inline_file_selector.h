/********************************************************************************
** Form generated from reading UI file 'inline_file_selector.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_INLINE_FILE_SELECTOR_H
#define UI_INLINE_FILE_SELECTOR_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_InlineFileSelector
{
public:
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *lineEdit;

    void setupUi(QWidget *InlineFileSelector)
    {
        if (InlineFileSelector->objectName().isEmpty())
            InlineFileSelector->setObjectName(QString::fromUtf8("InlineFileSelector"));
        InlineFileSelector->resize(400, 43);
        horizontalLayout = new QHBoxLayout(InlineFileSelector);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(InlineFileSelector);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        lineEdit = new QLineEdit(InlineFileSelector);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        horizontalLayout->addWidget(lineEdit);


        retranslateUi(InlineFileSelector);

        QMetaObject::connectSlotsByName(InlineFileSelector);
    } // setupUi

    void retranslateUi(QWidget *InlineFileSelector)
    {
        InlineFileSelector->setWindowTitle(QApplication::translate("InlineFileSelector", "Form", nullptr));
        label->setText(QApplication::translate("InlineFileSelector", "File:", nullptr));
    } // retranslateUi

};

namespace Ui {
    class InlineFileSelector: public Ui_InlineFileSelector {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_INLINE_FILE_SELECTOR_H
