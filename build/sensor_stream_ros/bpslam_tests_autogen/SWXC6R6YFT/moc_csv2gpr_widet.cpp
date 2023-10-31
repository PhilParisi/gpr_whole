/****************************************************************************
** Meta object code from reading C++ file 'csv2gpr_widet.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/sensor_stream_ros/ui/csv2gpr_widet.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'csv2gpr_widet.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_CSV2GPRWidet_t {
    QByteArrayData data[17];
    char stringdata0[218];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_CSV2GPRWidet_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_CSV2GPRWidet_t qt_meta_stringdata_CSV2GPRWidet = {
    {
QT_MOC_LITERAL(0, 0, 12), // "CSV2GPRWidet"
QT_MOC_LITERAL(1, 13, 23), // "on_open_csv_btn_clicked"
QT_MOC_LITERAL(2, 37, 0), // ""
QT_MOC_LITERAL(3, 38, 22), // "on_predict_btn_clicked"
QT_MOC_LITERAL(4, 61, 11), // "onMouseMove"
QT_MOC_LITERAL(5, 73, 12), // "QMouseEvent*"
QT_MOC_LITERAL(6, 86, 5), // "event"
QT_MOC_LITERAL(7, 92, 16), // "onPred1dFinished"
QT_MOC_LITERAL(8, 109, 15), // "gpr::Prediction"
QT_MOC_LITERAL(9, 125, 4), // "pred"
QT_MOC_LITERAL(10, 130, 23), // "on_generate_lml_clicked"
QT_MOC_LITERAL(11, 154, 14), // "updateLMLRange"
QT_MOC_LITERAL(12, 169, 3), // "min"
QT_MOC_LITERAL(13, 173, 3), // "max"
QT_MOC_LITERAL(14, 177, 19), // "updateLMLColorscale"
QT_MOC_LITERAL(15, 197, 15), // "setMaxBlockSize"
QT_MOC_LITERAL(16, 213, 4) // "size"

    },
    "CSV2GPRWidet\0on_open_csv_btn_clicked\0"
    "\0on_predict_btn_clicked\0onMouseMove\0"
    "QMouseEvent*\0event\0onPred1dFinished\0"
    "gpr::Prediction\0pred\0on_generate_lml_clicked\0"
    "updateLMLRange\0min\0max\0updateLMLColorscale\0"
    "setMaxBlockSize\0size"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_CSV2GPRWidet[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x0a /* Public */,
       3,    0,   55,    2, 0x0a /* Public */,
       4,    1,   56,    2, 0x0a /* Public */,
       7,    1,   59,    2, 0x0a /* Public */,
      10,    0,   62,    2, 0x0a /* Public */,
      11,    2,   63,    2, 0x0a /* Public */,
      14,    0,   68,    2, 0x0a /* Public */,
      15,    1,   69,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, 0x80000000 | 8,    9,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Double, QMetaType::Double,   12,   13,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   16,

       0        // eod
};

void CSV2GPRWidet::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<CSV2GPRWidet *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->on_open_csv_btn_clicked(); break;
        case 1: _t->on_predict_btn_clicked(); break;
        case 2: _t->onMouseMove((*reinterpret_cast< QMouseEvent*(*)>(_a[1]))); break;
        case 3: _t->onPred1dFinished((*reinterpret_cast< gpr::Prediction(*)>(_a[1]))); break;
        case 4: _t->on_generate_lml_clicked(); break;
        case 5: _t->updateLMLRange((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 6: _t->updateLMLColorscale(); break;
        case 7: _t->setMaxBlockSize((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject CSV2GPRWidet::staticMetaObject = { {
    &QWidget::staticMetaObject,
    qt_meta_stringdata_CSV2GPRWidet.data,
    qt_meta_data_CSV2GPRWidet,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *CSV2GPRWidet::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *CSV2GPRWidet::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CSV2GPRWidet.stringdata0))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int CSV2GPRWidet::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
