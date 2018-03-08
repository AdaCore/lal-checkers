package body Null_Deref is
   type Integer_Linked_List;

   type Integer_Linked_List_Access is access all Integer_Linked_List;

   type Integer_Linked_List is record
      Next : Integer_Linked_List_Access;
      Value : Integer;
   end record;

   function Foo(My_List : Integer_Linked_List_Access) return Integer is
      List_Cursor : Integer_Linked_List_Access := My_List;
      Found : Boolean := False;
   begin
      if My_List /= null then
          Find_Fourty_Two:
             while List_Cursor /= null loop
                if List_Cursor.Value = 42 then
                   Found := True;
                   exit Find_Fourty_Two;
                end if;
                List_Cursor := List_Cursor.Next;
             end loop Find_Fourty_Two;

          return (if Found then 42 else List_Cursor.Value);
      end if;
   end Foo;
end Null_Deref;
