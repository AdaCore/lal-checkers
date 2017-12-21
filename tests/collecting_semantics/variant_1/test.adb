procedure Ex1 is
   type My_Variant(b : Boolean) is record
      case b is
         when True =>
            x : Integer;
         when False =>
            y : Boolean;
      end case;
   end record;
   x : My_Variant := (b => False, y => True);
begin
   x.y := x.b;
end Ex1;
