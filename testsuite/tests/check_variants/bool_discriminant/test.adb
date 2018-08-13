procedure Test is
   type My_Variant(B : Boolean) is record
      case B is
         when True =>
            X : Integer;
         when False =>
            Y : Integer;
      end case;
   end record;

   x : My_Variant;
   res : Integer;
begin
   if x.X = 12 then
      res := 42;
   else
      res := x.Y;
   end if;
end Ex1;
