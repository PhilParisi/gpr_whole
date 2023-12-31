; Auto-generated. Do not edit!


(cl:in-package template_package_msgs-msg)


;//! \htmlinclude example.msg.html

(cl:defclass <example> (roslisp-msg-protocol:ros-message)
  ((first_name
    :reader first_name
    :initarg :first_name
    :type cl:string
    :initform "")
   (last_name
    :reader last_name
    :initarg :last_name
    :type cl:string
    :initform "")
   (age
    :reader age
    :initarg :age
    :type cl:fixnum
    :initform 0)
   (score
    :reader score
    :initarg :score
    :type cl:integer
    :initform 0))
)

(cl:defclass example (<example>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <example>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'example)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name template_package_msgs-msg:<example> is deprecated: use template_package_msgs-msg:example instead.")))

(cl:ensure-generic-function 'first_name-val :lambda-list '(m))
(cl:defmethod first_name-val ((m <example>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader template_package_msgs-msg:first_name-val is deprecated.  Use template_package_msgs-msg:first_name instead.")
  (first_name m))

(cl:ensure-generic-function 'last_name-val :lambda-list '(m))
(cl:defmethod last_name-val ((m <example>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader template_package_msgs-msg:last_name-val is deprecated.  Use template_package_msgs-msg:last_name instead.")
  (last_name m))

(cl:ensure-generic-function 'age-val :lambda-list '(m))
(cl:defmethod age-val ((m <example>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader template_package_msgs-msg:age-val is deprecated.  Use template_package_msgs-msg:age instead.")
  (age m))

(cl:ensure-generic-function 'score-val :lambda-list '(m))
(cl:defmethod score-val ((m <example>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader template_package_msgs-msg:score-val is deprecated.  Use template_package_msgs-msg:score instead.")
  (score m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <example>) ostream)
  "Serializes a message object of type '<example>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'first_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'first_name))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'last_name))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'last_name))
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'age)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'score)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'score)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'score)) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'score)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <example>) istream)
  "Deserializes a message object of type '<example>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'first_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'first_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'last_name) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'last_name) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'age)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'score)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:slot-value msg 'score)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:slot-value msg 'score)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:slot-value msg 'score)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<example>)))
  "Returns string type for a message object of type '<example>"
  "template_package_msgs/example")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'example)))
  "Returns string type for a message object of type 'example"
  "template_package_msgs/example")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<example>)))
  "Returns md5sum for a message object of type '<example>"
  "f8bfa80ae3c7a93455596d9622ad33a9")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'example)))
  "Returns md5sum for a message object of type 'example"
  "f8bfa80ae3c7a93455596d9622ad33a9")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<example>)))
  "Returns full string definition for message of type '<example>"
  (cl:format cl:nil "string first_name~%string last_name~%uint8 age~%uint32 score~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'example)))
  "Returns full string definition for message of type 'example"
  (cl:format cl:nil "string first_name~%string last_name~%uint8 age~%uint32 score~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <example>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'first_name))
     4 (cl:length (cl:slot-value msg 'last_name))
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <example>))
  "Converts a ROS message object to a list"
  (cl:list 'example
    (cl:cons ':first_name (first_name msg))
    (cl:cons ':last_name (last_name msg))
    (cl:cons ':age (age msg))
    (cl:cons ':score (score msg))
))
